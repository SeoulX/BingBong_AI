$(document).ready(function () {
  var conversation = [];
  var conversationID = null;
  var loggedInUsername = $("#logged-in-username").val();

  console.log(loggedInUsername)

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
    conversation = [];
    conversationID = null;
    console.log("Conversation: ", conversation)

    // CreateNew();

    $("#chat-messages").empty();
  });

  function sendMessage() {
    var userMessage = $("#user-input").val().trim();
    if (userMessage === "") return;

      $("#chat-messages").append(
        '<p class="user-message"> ' + userMessage + "</p>"
      );
      conversation.push({
        sender: loggedInUsername,
        message: userMessage,
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
              sender: "Bingbong",
              message: response.response,
            });
            scrollToBottom();
            $.ajax({
                      url: "/save_conversation/",
                      type: "POST",
                      data: {
                        message: userMessage,
                        csrfmiddlewaretoken: "{{ csrf_token }}",
                        conversation: JSON.stringify(conversation),
                        conversationID: conversationID,
                      },
                      success: function (response) {
                        if (!conversationID) {
                          conversationID = response.conversation_id;
                          console.log(conversationID);
                        }
                      },
                      error: function (xhr, errmsg, err) {
                        console.error("Error:", errmsg);
                      }
                    });
          },
          error: function (xhr, errmsg, err) {
            console.log(errmsg);
            $(".typing-indicator").remove();
          },
        });
      }, 1100);
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

