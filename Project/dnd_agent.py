from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_core.language_models import BaseChatModel
from langchain.tools import BaseTool, tool, ToolRuntime
from langchain_text_splitters import TextSplitter
from langgraph.graph.state import CompiledStateGraph
from typing import Any, List, Optional
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.vectorstores import InMemoryVectorStore

from dotenv import load_dotenv
import os

# Convenience typedefs
AgentType = CompiledStateGraph[Any, Any, Any, Any]

#
# Context & state class definitions
#

from pydantic import BaseModel
from langchain.agents import AgentState

class DnDContext(BaseModel):
    """Runtime context for the D&D assistant containing game information."""
    game_master_name: str
    player_level: int = 1
    campaign_setting: str = "Forgotten Realms"

class DnDAgentState(AgentState):
    """Custom state for the D&D assistant."""
    rules_reference_used: List[str] = []
    current_topic: str = ""


class DnDAssistant:
    """Wrapper class for the D&D assistant to use in UI."""
    
    def __init__(self, game_master_name="Alex", player_level=3, campaign_setting="Forgotten Realms"):
        load_dotenv()
        self.agent = self._setup_assistant()
        self.config = {"configurable": {"thread_id": "1"}}
        self.context = DnDContext(
            game_master_name=game_master_name,
            player_level=player_level,
            campaign_setting=campaign_setting
        )
        self._referenced_rules = []  # Nowa lista do przechowywania reguł
    
    def _load_rules_documents(self, document_loader: BaseLoader, text_splitter: TextSplitter) -> List[Document]:
        try:
            all_pages = document_loader.load()
            if not all_pages:
                return []
            return text_splitter.split_documents(all_pages)
        except Exception:
            return []
    
    def _prepare_vector_store(self, rules_chunks: List[Document], embeddings: Embeddings) -> VectorStore:
        return InMemoryVectorStore.from_documents(
            documents=rules_chunks,
            embedding=embeddings,
        )
    
    def _setup_assistant(self) -> AgentType:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize model
        model = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_retries=2,
            timeout=30.0,
            max_tokens=512
        )
        
        # Load PDF
        pdf_path = "Player's Handbook.pdf"
        if not os.path.exists(pdf_path):
            # Create dummy documents if PDF not found
            rules_docs = [
                Document(page_content="Goliath: A race of powerful humanoids known for their strength and size.", metadata={"page": 123})
            ]
        else:
            loader = PyPDFLoader(pdf_path)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                add_start_index=True
            )
            rules_docs = self._load_rules_documents(loader, splitter)
        
        # Create vector store
        rules_vector_store = self._prepare_vector_store(rules_docs, embeddings)
        
        @tool
        def search_rules(query: str) -> str:
            """Search the D&D 5e rules for specific information."""
            try:
                docs = rules_vector_store.similarity_search(query, k=4)
                if not docs:
                    return "No matching rules found in the rulebooks."
                
                # Dodaj zapytanie do listy reguł
                self._add_referenced_rule(f"Search: {query}")
                
                # Dodaj tematy z dokumentów (uproszczone)
                for doc in docs[:2]:  # Pierwsze 2 dokumenty
                    content_lower = doc.page_content.lower()
                    if any(keyword in content_lower for keyword in ['race', 'class', 'spell', 'item', 'monster', 'rule']):
                        # Wyciągnij pierwsze słowo jako potencjalną nazwę
                        words = doc.page_content.split()[:3]
                        if words:
                            rule_name = " ".join(words).strip('.,:;')
                            self._add_referenced_rule(f"Found: {rule_name}")
                
                content = "\n\n".join((f"Source: Page {doc.metadata.get('page', 'N/A')}\n{doc.page_content}") 
                                     for doc in docs)
                return content
            except Exception as e:
                return f"Error searching rules: {str(e)}"
        
        # Usuń narzędzie log_rules_reference - będziemy używać własnej listy
        tools: List[BaseTool] = [search_rules]
        
        # Bind tools to model
        model_with_tools = model.bind_tools(tools)
        
        prompt: str = (
            "You are a helpful Dungeons & Dragons 5th Edition assistant. Your purpose is to help "
            "players and Dungeon Masters understand game rules, mechanics, and provide guidance "
            "on various aspects of the game.\n\n"
            
            "Available tools:\n"
            "- search_rules: Search the D&D 5e rulebooks for specific rules, spells, items, or mechanics\n\n"
            
            "IMPORTANT INSTRUCTIONS:\n"
            "1. When a user asks about ANY rule, race, class, spell, item, or mechanic, you MUST use search_rules tool first\n"
            "2. Always mention the specific names of rules, races, classes, spells, items, or monsters in your answers\n"
            "3. Be specific and accurate with rule names\n\n"
            
            "Context information:\n"
            "Campaign Setting: {campaign_setting}\n"
            "Player Level: {player_level}\n"
            "Game Master: {game_master_name}\n\n"
            
            "Begin by greeting the user and asking how you can help with their D&D game."
        )

        checkpointer = MemorySaver()
        
        agent = create_agent(
            model=model_with_tools,
            tools=tools,
            state_schema=DnDAgentState,
            context_schema=DnDContext,
            system_prompt=prompt,
            checkpointer=checkpointer,
        )
        
        return agent
    


    def _add_referenced_rule(self, rule: str):
        """Add a rule to the referenced rules list."""
        if rule and rule not in self._referenced_rules:
            self._referenced_rules.append(rule)
    
    
    def send_message(self, message: str) -> str:
        """Send a message to the assistant and get response."""
        try:
            # Dodaj zapytanie użytkownika do listy reguł
            self._add_referenced_rule(f"Query: {message}")
            
            # Get the response from agent
            events = list(self.agent.stream(
                {"messages": [{"role": "user", "content": message}]},
                config=self.config,
                context=self.context,
                stream_mode="values",
            ))
            
            # Extract the last message
            response_text = "Sorry, I couldn't generate a response."
            if events:
                last_event = events[-1]
                if "messages" in last_event and last_event["messages"]:
                    last_message = last_event["messages"][-1]
                    if hasattr(last_message, 'content'):
                        response_text = last_message.content
            
            # Dodaj odpowiedź do analizy (opcjonalnie)
            self._extract_rules_from_response(response_text, message)
            
            return response_text
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _extract_rules_from_response(self, response: str, query: str):
        """Extract potential rule references from response."""
        # Prosta heurystyka: szukaj dużych liter na początku wyrazów
        words = response.split()
        for i, word in enumerate(words):
            # Jeśli słowo zaczyna się z dużej litery i ma co najmniej 3 znaki
            if word and word[0].isupper() and len(word) > 2 and word.isalpha():
                # Sprawdź czy to może być nazwa (nie na początku zdania)
                if i > 0 and words[i-1][-1] not in '.!?':
                    self._add_referenced_rule(f"Possible: {word}")
    
    def get_referenced_rules(self) -> List[str]:
        """Get list of referenced rules."""
        return self._referenced_rules.copy()