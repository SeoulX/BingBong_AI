// script.js

$(document).ready(function () {
  var conversationCount = 0;
  var currentConversation = [];
  var conversationStarted = false;

  $("#send-btn").click(function () {
    sendMessage();
  });

  $("#user-input").keypress(function (event) {
    if (event.key === "Enter") {
      event.preventDefault(); // Prevent default form submission
      sendMessage();
    }
  });

  $("#hide-history-btn").click(function () {
    $("#history-panel").hide();
    $("#show-history-btn").show();
    adjustChatInputPosition();
  });

  $("#show-history-btn").click(function () {
    $("#history-panel").show();
    $("#show-history-btn").hide();
    adjustChatInputPosition();
  });

  $("#add-conversation-btn").click(function () {
    addToHistory(currentConversation);
    resetConversation();
  });

  function resetConversation() {
    // Clear chat messages
    $("#chat-messages").empty();
    currentConversation = [];
    conversationStarted = false; // Reset the flag for new conversation
    conversationCount++;
    localStorage.removeItem("conversation-" + conversationCount);
  }

  function sendMessage() {
    var userMessage = $("#user-input").val().trim();
    if (userMessage === "") return;

    if (!conversationStarted) {
      conversationCount++;
      addToHistory(currentConversation); // Add new conversation to history
      conversationStarted = true;
    }

    $("#chat-messages").append(
      '<p class="user-message"> ' + userMessage + "</p>"
    );
    currentConversation.push({ type: "user", message: userMessage });

    // Clear the input field
    $("#user-input").val("");

    scrollToBottom();

    // Show typing indicator with a delay
    setTimeout(function () {
      $("#chat-messages").append(
        '<p class="bot-message typing-indicator"><strong></strong> <span class="typing-dots"><span>•</span><span>•</span><span>•</span></span></p>'
      );
      scrollToBottom();
    }, 500); // Delay of 500ms (0.5 seconds)

    // Delay bot response by 5 seconds
    setTimeout(function () {
      $.ajax({
        url: "/process_message/",
        type: "POST",
        data: {
          message: userMessage,
          csrfmiddlewaretoken: "{{ csrf_token }}",
        },
        success: function (response) {
          $(".typing-indicator").remove();
          $("#chat-messages").append(
            '<p class="bot-message"><strong>Bingbong:</strong> ' +
              response.response +
              "</p>"
          );
          currentConversation.push({
            type: "bot",
            message: response.response,
          });
          scrollToBottom(); // Scroll to bottom after receiving response
          updateHistory(); // Update history with new message
        },
        error: function (xhr, errmsg, err) {
          console.log(errmsg);
          $(".typing-indicator").remove();
        },
      });
    }, 1100); // 1.1 seconds delay for bot response (500ms for typing indicator + 0.6 seconds delay)
  }

  function addToHistory(conversation) {
    var historyItem = $(
      '<li class="chat-history-item" data-conversation-id="' +
        conversationCount +
        '">Conversation ' +
        conversationCount +
        "</li>"
    );
    historyItem.click(function () {
      loadConversation($(this).data("conversation-id"));
    });
    $("#chat-history-list").append(historyItem);

    saveConversation(conversationCount, conversation);
  }

  function saveConversation(conversationId, conversation) {
    localStorage.setItem(
      "conversation-" + conversationId,
      JSON.stringify(conversation)
    );
  }

  function loadConversation(conversationId) {
    var conversation = JSON.parse(
      localStorage.getItem("conversation-" + conversationId)
    );
    if (conversation) {
      $("#chat-messages").empty();
      conversation.forEach(function (msg) {
        if (msg.type === "user") {
          $("#chat-messages").append(
            '<p class="user-message"> ' + msg.message + "</p>"
          );
        } else if (msg.type === "bot") {
          $("#chat-messages").append(
            '<p class="bot-message"><strong>Bingbong:</strong> ' +
              msg.message +
              "</p>"
          );
        }
      });
      scrollToBottom();
    }
  }

  function updateHistory() {
    saveConversation(conversationCount, currentConversation);
  }

  function scrollToBottom() {
    console.log("Scrolling to bottom...");
    $("#chat-messages").animate(
      { scrollTop: $("#chat-messages").prop("scrollHeight") },
      500,
      function() {
        console.log("Height of chat-container" , $("#chat-messages").prop("scrollHeight"));
      }
    );
  }

  function adjustChatInputPosition() {
    if ($("#history-panel").is(":visible")) {
      $(".chat-input").css("left", "calc(25% + 10px)");
    } else {
      $(".chat-input").css("left", "10px");
    }
  }

  $("#show-history-btn").hide();
  adjustChatInputPosition();
});
