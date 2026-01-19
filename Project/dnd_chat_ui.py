import sys
import threading
import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QScrollArea, QFrame,
    QSplitter, QListWidget, QListWidgetItem, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, pyqtSlot
from PyQt5.QtGui import QFont, QTextCursor, QColor, QPalette, QKeyEvent

from dnd_agent import DnDAssistant


class ChatWorker(QThread):
    """Worker thread for handling AI responses."""
    response_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, assistant, message):
        super().__init__()
        self.assistant = assistant
        self.message = message
    
    def run(self):
        try:
            response = self.assistant.send_message(self.message)
            self.response_ready.emit(response)
        except Exception as e:
            self.error_occurred.emit(str(e))


class DnDChatWindow(QMainWindow):
    """Main chat window for D&D Assistant."""
    
    def __init__(self):
        super().__init__()
        self.assistant = None
        self.is_processing = False
        self.init_ui()
        self.init_assistant()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("D&D 5e Assistant")
        self.setGeometry(100, 100, 900, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for chat and rules panel
        splitter = QSplitter(Qt.Horizontal)
        
        # Chat area (left side)
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        
        # Chat header
        header_label = QLabel("D&D 5e Assistant - Forgotten Realms")
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        header_label.setStyleSheet("""
            QLabel {
                color: #8B4513;
                padding: 10px;
                background-color: #F5DEB3;
                border-radius: 5px;
            }
        """)
        header_label.setAlignment(Qt.AlignCenter)
        chat_layout.addWidget(header_label)
        
        # Chat display area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Arial", 11))
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #FAF0E6;
                border: 2px solid #8B4513;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        chat_layout.addWidget(self.chat_display)
        
        # Input area
        input_widget = QWidget()
        input_layout = QHBoxLayout(input_widget)
        input_layout.setContentsMargins(0, 5, 0, 0)
        
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Type your D&D question here... (Press Enter to send)")
        self.message_input.setFont(QFont("Arial", 11))
        self.message_input.setStyleSheet("""
            QLineEdit {
                padding: 12px;
                border: 2px solid #8B4513;
                border-radius: 8px;
                background-color: white;
                selection-background-color: #DEB887;
            }
            QLineEdit:focus {
                border: 2px solid #D2691E;
                background-color: #FFF8DC;
            }
            QLineEdit:disabled {
                background-color: #F5F5F5;
                border: 2px solid #CCCCCC;
                color: #888888;
            }
        """)
        self.message_input.returnPressed.connect(self.send_message)
        self.message_input.setFocusPolicy(Qt.StrongFocus)
        
        self.send_button = QPushButton("Send")
        self.send_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.send_button.setFixedWidth(100)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #8B4513;
                color: white;
                padding: 12px 5px;
                border-radius: 8px;
                border: none;
                margin-left: 10px;
            }
            QPushButton:hover {
                background-color: #A0522D;
            }
            QPushButton:pressed {
                background-color: #654321;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        self.send_button.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.message_input, 1)  # 1 = stretch factor
        input_layout.addWidget(self.send_button)
        
        chat_layout.addWidget(input_widget)
        
        # Rules panel (right side)
        rules_widget = QWidget()
        rules_layout = QVBoxLayout(rules_widget)
        
        rules_header = QLabel("Referenced Rules")
        rules_header.setFont(QFont("Arial", 12, QFont.Bold))
        rules_header.setStyleSheet("""
            QLabel {
                color: #8B4513;
                padding: 10px;
                background-color: #DEB887;
                border-radius: 5px;
            }
        """)
        rules_header.setAlignment(Qt.AlignCenter)
        rules_layout.addWidget(rules_header)
        
        self.rules_list = QListWidget()
        self.rules_list.setFont(QFont("Arial", 10))
        self.rules_list.setStyleSheet("""
            QListWidget {
                background-color: #FFF8DC;
                border: 2px solid #8B4513;
                border-radius: 10px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #DEB887;
            }
            QListWidget::item:selected {
                background-color: #DEB887;
            }
        """)
        rules_layout.addWidget(self.rules_list)
        
        # Clear rules button
        clear_rules_btn = QPushButton("Clear Rules List")
        clear_rules_btn.setFont(QFont("Arial", 10))
        clear_rules_btn.setStyleSheet("""
            QPushButton {
                background-color: #CD853F;
                color: white;
                padding: 8px;
                border-radius: 5px;
                border: none;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #D2691E;
            }
        """)
        clear_rules_btn.clicked.connect(self.clear_rules_list)
        rules_layout.addWidget(clear_rules_btn)
        
        # Add widgets to splitter
        splitter.addWidget(chat_widget)
        splitter.addWidget(rules_widget)
        splitter.setSizes([600, 300])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Add initial welcome message
        self.add_message("System", "Welcome to D&D 5e Assistant! How can I help you with your adventure today?", is_user=False)
        
        # Set focus to message input after a short delay
        QTimer.singleShot(100, self.focus_message_input)
    
    def focus_message_input(self):
        """Set focus to message input field."""
        if self.message_input.isEnabled():
            self.message_input.setFocus()
    
    def init_assistant(self):
        """Initialize the D&D assistant."""
        self.statusBar().showMessage("Initializing D&D Assistant...")
        self.set_input_enabled(False)
        
        # Initialize assistant in a separate thread to avoid UI freezing
        def init_assistant_thread():
            try:
                self.assistant = DnDAssistant(
                    game_master_name="Alex",
                    player_level=3,
                    campaign_setting="Forgotten Realms"
                )
                # Schedule UI update on main thread
                QTimer.singleShot(0, self.on_assistant_ready)
            except Exception as e:
                QTimer.singleShot(0, lambda: self.on_assistant_error(str(e)))
        
        threading.Thread(target=init_assistant_thread, daemon=True).start()
    
    def set_input_enabled(self, enabled):
        """Enable or disable input controls."""
        self.send_button.setEnabled(enabled)
        self.message_input.setEnabled(enabled)
        if enabled:
            self.message_input.setFocus()
    
    def on_assistant_ready(self):
        """Called when assistant is ready."""
        self.statusBar().showMessage("Assistant ready!")
        self.set_input_enabled(True)
    
    def on_assistant_error(self, error):
        """Called when assistant initialization fails."""
        self.statusBar().showMessage(f"Error: {error}")
        self.add_message("System", f"Failed to initialize assistant: {error}", is_user=False, is_error=True)
        self.set_input_enabled(False)
        
        QMessageBox.critical(
            self,
            "Initialization Error",
            f"Failed to initialize D&D Assistant:\n\n{error}\n\nPlease check your API key and PDF file."
        )
    
    def get_current_time(self):
        """Get current time formatted for display."""
        return datetime.datetime.now().strftime("%H:%M")
    
    def add_message(self, sender, message, is_user=True, is_error=False):
        """Add a message to the chat display."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Format the message
        timestamp = self.get_current_time()
        
        if is_error:
            color = "#FF0000"
            prefix = "❌ ERROR"
        elif is_user:
            color = "#2E8B57"  # Sea Green
            prefix = "🧙 You"
        else:
            color = "#8B4513"  # Saddle Brown
            if sender == "System":
                prefix = "⚙️ System"
            else:
                prefix = "⚔️ Assistant"
        
        formatted_message = f"""
        <div style="margin: 10px 0; padding: 10px; border-radius: 10px; background-color: {'#E6F3E6' if is_user else '#F5F5DC'}; border-left: 5px solid {color};">
            <div style="font-weight: bold; color: {color}; margin-bottom: 5px;">
                {prefix} • {timestamp}
            </div>
            <div style="color: #333;">
                {message.replace('\n', '<br>')}
            </div>
        </div>
        """
        
        self.chat_display.append(formatted_message)
        
        # Scroll to bottom
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def update_rules_list(self):
        """Update the rules list from the assistant."""
        if self.assistant:
            rules = self.assistant.get_referenced_rules()
            self.rules_list.clear()
            for rule in rules:
                item = QListWidgetItem(f"📖 {rule}")
                self.rules_list.addItem(item)
    
    def clear_rules_list(self):
        """Clear the rules list."""
        self.rules_list.clear()
        self.add_message("System", "Rules list cleared.", is_user=False)
    
    def send_message(self):
        """Send a message to the assistant."""
        message = self.message_input.text().strip()
        if not message or self.is_processing:
            return
        
        # Clear input
        self.message_input.clear()
        
        # Set processing flag
        self.is_processing = True
        self.set_input_enabled(False)
        
        # Add user message to chat
        self.add_message("You", message, is_user=True)
        self.statusBar().showMessage("Processing...")
        
        # Create worker thread for AI response
        self.worker = ChatWorker(self.assistant, message)
        self.worker.response_ready.connect(self.on_response_received)
        self.worker.error_occurred.connect(self.on_response_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()
    
    @pyqtSlot()
    def on_worker_finished(self):
        """Called when worker thread finishes."""
        self.worker = None
    
    def on_response_received(self, response):
        """Handle AI response."""
        self.add_message("Assistant", response, is_user=False)
        self.statusBar().showMessage("Ready")
        
        # Update rules list
        self.update_rules_list()
        
        # Reset processing flag and enable input
        self.is_processing = False
        self.set_input_enabled(True)
    
    def on_response_error(self, error):
        """Handle AI response error."""
        self.add_message("System", f"Error: {error}", is_user=False, is_error=True)
        self.statusBar().showMessage(f"Error: {error}")
        
        # Reset processing flag and enable input
        self.is_processing = False
        self.set_input_enabled(True)
    
    def keyPressEvent(self, event):
        """Handle key press events."""
        # Focus message input when typing anywhere in window
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            # Already handled by returnPressed signal
            pass
        elif not event.modifiers() and event.text().isprintable():
            # If printable character pressed and message input is enabled, focus it
            if self.message_input.isEnabled():
                self.message_input.setFocus()
                # Forward the key press to the message input
                self.message_input.keyPressEvent(event)
                return
        
        super().keyPressEvent(event)
    
    def showEvent(self, event):
        """Handle show event to set focus."""
        super().showEvent(event)
        # Set focus to message input when window is shown
        QTimer.singleShot(100, self.focus_message_input)
    
    def closeEvent(self, event):
        """Handle window close event."""
        reply = QMessageBox.question(
            self,
            'Exit',
            'Are you sure you want to exit?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application style
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(253, 245, 230))
    palette.setColor(QPalette.WindowText, Qt.darkGray)
    palette.setColor(QPalette.Base, QColor(255, 250, 240))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 220))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.black)
    palette.setColor(QPalette.Button, QColor(245, 222, 179))
    palette.setColor(QPalette.ButtonText, Qt.black)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(139, 69, 19))
    palette.setColor(QPalette.Highlight, QColor(139, 69, 19))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(palette)
    
    window = DnDChatWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()