from logging import config
import streamlit as st
from llm_chains import load_normal_chain
from langchain.memory import StreamlitChatMessageHistory
from streamlit_mic_recorder import mic_recorder
from utils import save_chat_history_json, get_timestamp, load_chat_history_json
from audio_handler import transcribe_audio
import yaml
import os
from html_templates import css

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_chain(chat_history):
    return load_normal_chain(chat_history)

def set_send_input():
    st.session_state.user_question = st.session_state.temp_input  
    st.session_state.temp_input = ""  
    st.session_state.send_input = True

def save_chat_history():
    if st.session_state.history != []:
        if st.session_state.session_key == "new_session":
            st.session_state.new_session_key = get_timestamp() + ".json"
            save_chat_history_json(st.session_state.history, config["chat_history_path"] + st.session_state.new_session_key)
        else:
            save_chat_history_json(st.session_state.history, config["chat_history_path"] + st.session_state.session_key)

def main():
    st.title("Chat with AI")
    st.write(css, unsafe_allow_html=True)
    chat_container = st.container()
    st.sidebar.title("Chat Sessions")
    chat_sessions = ["new_session"] + os.listdir(config["chat_history_path"])
    
    if "send_input" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.send_input = False
        st.session_state.user_question = ""
        st.session_state.temp_input = ""  
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"

    if st.session_state.session_key == "new_session" and st.session_state.new_session_key is not None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox("Select a chat session", chat_sessions, key="session_key", index=index)
    
    if st.session_state.session_key != "new_session":
        st.session_state.history = load_chat_history_json(config["chat_history_path"] + st.session_state.session_key)
    else:
        st.session_state.history = []

    chat_history = StreamlitChatMessageHistory(key="history")
    llm_chain = load_chain(chat_history)

    user_input = st.text_input("Type your message here", key="temp_input")

    voice_recording_column, send_button_column = st.columns(2)
    with voice_recording_column:
        voice_recording = mic_recorder(start_prompt="Start recording", stop_prompt="Stop recording", just_once=False)
    with send_button_column:
        send_button = st.button("Send", key="send_button", on_click=set_send_input)

    uploaded_audio = st.sidebar.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

    if uploaded_audio:
        transcribed_audio = transcribe_audio(uploaded_audio.getvalue())
        print(transcribed_audio)
        llm_chain = load_chain(chat_history)
        llm_chain.run("Summarize this text: " + transcribed_audio)
    
    if voice_recording:
        transcribed_audio = transcribe_audio(voice_recording["bytes"])
        print(transcribed_audio)
        llm_chain = load_chain(chat_history)
        llm_chain.run(transcribed_audio)
    
    if send_button or st.session_state.send_input or voice_recording:
        
        if st.session_state.user_question:  
            with chat_container:
                st.chat_message("user").write(st.session_state.user_question)
                llm_response = llm_chain.run(st.session_state.user_question)
                st.chat_message("assistant").write(llm_response)
                st.session_state.user_question = ""  
        
        if voice_recording:
            with chat_container:
                st.chat_message("user").write(transcribed_audio)
                llm_response = llm_chain.run(transcribed_audio)
                st.chat_message("assistant").write(llm_response)

        st.session_state.send_input = False  

    if chat_history.messages != []:
        with chat_container:
            st.write("Chat History:")
            for message in chat_history.messages:
                st.chat_message(message.type).write(message.content)

    save_chat_history()

if __name__ == "__main__":
    main()
