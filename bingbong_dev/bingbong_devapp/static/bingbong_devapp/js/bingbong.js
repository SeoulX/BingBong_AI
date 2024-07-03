$(document).ready(function () {
  var conversation = [];

  $("#send-btn").click(sendMessage);
  $("#user-input").keypress(function (event) {
    if (event.key === "Enter") {
      event.preventDefault();
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
    console.log("Conversation:", conversation);
  });

  function sendMessage() {
    var userMessage = $("#user-input").val().trim();
    if (userMessage === "") return;

    $("#chat-messages").append(
      '<p class="user-message"> ' + userMessage + "</p>"
    );
    conversation.push({
      sender: "user",
      message: userMessage,
      timestamp: new Date(), 
    });

    $("#user-input").val("");

    scrollToBottom();

    setTimeout(function () {
      $("#chat-messages").append(
        '<p class="bot-message typing-indicator"><strong></strong> <span class="typing-dots"><span>•</span><span>•</span><span>•</span></span></p>'
      );
      scrollToBottom();
    }, 500); 

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
          conversation.push({
            sender: "bot",
            message: response.response,
            timestamp: new Date(),
          });
          scrollToBottom(); 
        },
        error: function (xhr, errmsg, err) {
          console.log(errmsg);
          $(".typing-indicator").remove();
        },
      });
    }, 1100);
    console.log("Conversation:", conversation);
  }

  function scrollToBottom() {
    $("#chat-messages").animate({ scrollTop: $("#chat-messages").prop("scrollHeight") }, 500);
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

