const messageInput = document.getElementById('message');
const sendBtn = document.getElementById('send-btn');
const chatHistory = document.querySelector('.chat-history');

sendBtn.addEventListener('click', sendMessage);

function sendMessage() {
  const message = messageInput.value.trim();
  if (message) {
    // Create message element (list item)
    const messageElement = document.createElement('li'); // Changed from 'p' to 'li' for consistency with HTML structure
    messageElement.innerText = message;
    messageElement.classList.add('user-message');

    // **Crucial Fix: Append user message first**
    chatHistory.appendChild(messageElement); 

    // Clear input field
    messageInput.value = '';

    // Simulate bot response (replace with API call)
    const botResponse = "I'm getting smarter, ask me anything!"; // Replace with API response
    appendMessage(botResponse, 'bot-message');
  }
}

function appendMessage(message, className) {
  const messageElement = document.createElement('li'); // Changed from 'p' to 'li' for consistency with HTML structure
  messageElement.innerText = message;
  messageElement.classList.add(className);
  chatHistory.appendChild(messageElement);  // Append the message to chat history
}
