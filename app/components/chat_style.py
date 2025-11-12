def load_chat_css():
    """Returns CSS styling for the EVE AI Chatbox."""
    return """
    <style>
    .chat-wrapper {
        max-width: 900px;
        margin: auto;
        padding: 15px;
    }
    .user-msg {
        background-color: #2C2F35;
        color: #EAEAEA;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 8px 0;
        text-align: right;
        width: 85%;
        margin-left: auto;
    }
    .assistant-msg {
        background-color: #1E1F24;
        color: #EAEAEA;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 8px 0;
        width: 85%;
        margin-right: auto;
        border-left: 4px solid #4CC9F0;
    }
    .assistant-title {
        color: #4CC9F0;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .stChatInputContainer {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    .floating-clear-btn {
        position: fixed;
        bottom: 20px;
        right: 35px;
        background-color: #4CC9F0;
        color: white;
        border: none;
        border-radius: 50px;
        padding: 10px 18px;
        font-weight: 500;
        cursor: pointer;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.3);
        transition: 0.2s;
    }
    .floating-clear-btn:hover {
        background-color: #3BA9D0;
    }
    </style>
    """
