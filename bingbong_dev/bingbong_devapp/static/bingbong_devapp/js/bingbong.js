$(document).ready(function () {
  var conversation = [];
  var conversationID = null;
  var loggedInUsername = $("#logged-in-username").val();

  loadConversationHistory();

  $('#user-input').prop('disabled', false);
  $('#send-btn').prop('disabled', false);

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

    $('#user-input').prop('disabled', false);
    $('#send-btn').prop('disabled', false);

    $("#chat-messages").empty();
  });

  function loadConversationHistory() {
    $.ajax({
        url: '/get_conversations/',
        type: 'GET',
        data: {
            username: loggedInUsername
        },
        success: function(data) {
            if (data.conversations && data.conversations.length > 0) {
                updateChatHistoryList(data.conversations);
            } else {
                $('#chat-history-list').append('<li>No past conversations.</li>');
            }
        },
        error: function(xhr, errmsg, err) {
            console.error("Error fetching conversations:", errmsg);
        }
    });
  }

  function updateChatHistoryList(conversations) {
      var chatHistoryList = $('#chat-history-list');
      chatHistoryList.empty();
      conversations.reverse();

      for (let i = 0; i < conversations.length; i++) {
          let convo = conversations[i];
          let listItem = $('<li class="btn-grad">');
          let button = $('<button class="btn-history">').text(convo.topic).attr('data-conversation-id', convo.conversation_id);
          updateButtonColorExist(listItem, convo.sentiment);
          
          button.click(function () {
              loadConversation(convo.conversation_id);
          });
          listItem.append(button); 
          if (i === 0) {
            console.log("Sentiment: ", convo.sentiment)
            if(convo.sentiment === null){
              listItem.appendTo(chatHistoryList); 
              listItem.css({ 
              transform: "scale(1)",
                opacity: 0,
                display: 'block' 
              });
              setTimeout(function() { 
              listItem.animate({ 
                transform: "scale(0.8)",
                opacity: 1 }, 
                500, "swing");
              }, 10);
            }else{
              updateButtonColor(listItem, convo.sentiment)
            }
        } else {
            listItem.append(button); 
            chatHistoryList.append(listItem);
        }
      }
  }
  function updateButtonColorExist(button, sentiment) {
    var buttonColor = getSentimentColor(sentiment);
    button.css('background-image', `linear-gradient(to right, rgb(199,199,169) 0%, #fff 36%, ${buttonColor.end} 46%)`);
  }
  function updateButtonColor(button, sentiment) {
    var buttonColor = getSentimentColor(sentiment);
    button.css('background-image', `linear-gradient(to right, rgb(199,199,169) 0%, #fff 36%, ${buttonColor.end} 65%)`);
    setTimeout(() => {
      button.css('background-image', `linear-gradient(to right, rgb(199,199,169) 0%, #fff 36%, ${buttonColor.end} 55%)`);
    }, 1000);
    setTimeout(() => {
      button.css('background-image', `linear-gradient(to right, rgb(199,199,169) 0%, #fff 36%, ${buttonColor.end} 46%)`);
    }, 1000);
  }

  function loadConversation(conversationId) {
    $.ajax({
        url: '/get_conversation_details/',
        type: 'GET',
        data: { conversation_id: conversationId },
        success: function(data) {
            conversation = data.messages;
            conversationID = conversationId;
            updateChatDisplay();
        },
        error: function(xhr, errmsg, err) {
            console.error("Error loading conversation:", errmsg);
        }
    });
    $('#user-input').prop('disabled', true);
    $('#send-btn').prop('disabled', true);
  }

  function updateChatDisplay() {
      $("#chat-messages").empty();
      conversation.forEach(function(message) {
          var messageClass = message.sender === loggedInUsername ? "user-message" : "bot-message";
          var messageHtml = '<p class="' + messageClass + '">' + message.message + '</p>';
          $("#chat-messages").append(messageHtml);
      });
      scrollToBottom();
  }
  function getSentimentColor(sentiment) {
    if (sentiment === null) {
        return { start: '#555555', end: '#8b8989' };
    } else if (sentiment > 0) {
        return { start: '#4CAF50', end: '#d2e6b5' };
    } else if (sentiment = 0) {
        return { start: '#FFC107', end: '#ffffb7' };
    } else {
        return { start: '#F44336', end: '#f4c1c1' };
    }
  }

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
              '<p class="bot-message"><strong>BingBong:</strong> ' +
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
                        loadConversationHistory();
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

