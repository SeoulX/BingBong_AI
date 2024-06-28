const messageInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const chatHistory = document.querySelector('.chat-history');

sendBtn.addEventListener('click', sendMessage);

function sendMessage() {
  const message = messageInput.value.trim();
  if (message) {
    // Create user message element
    appendMessage(message, 'user-message');

    // Clear input field
    messageInput.value = '';

    // Send message to server for processing
    sendToServer(message);
  }
}

function sendToServer(message) {
  $.ajax({
    url: "/process_message/",
    type: "POST",
    data: {
      message: message,
      csrfmiddlewaretoken: "{{ csrf_token }}",
    },
    success: function (response) {
      // Display bot's response
      appendMessage(response.response, 'bot-message');
      scrollToBottom();
    },
    error: function (xhr, errmsg, err) {
      console.log(errmsg);
    },
  });
}

function appendMessage(message, className) {
  const messageElement = document.createElement('li');
  messageElement.innerText = message;
  messageElement.classList.add(className);
  chatHistory.appendChild(messageElement);
}

function scrollToBottom() {
  chatHistory.scrollTop = chatHistory.scrollHeight;
}
