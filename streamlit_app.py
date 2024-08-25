import streamlit as st
import uuid
import json
from chatbot import Chatbot

class ChatbotApp:
    def __init__(self):
        self.chatbot = None
        self.setup_streamlit()

    def setup_streamlit(self):
        st.set_page_config(page_title="Chat with Files", layout="wide")
        st.header("Chat with Files and URLs using Gemini")

        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        session_id = st.session_state.session_id

        if 'conversations' not in st.session_state:
            st.session_state.conversations = {}
        if session_id not in st.session_state.conversations:
            st.session_state.conversations[session_id] = {'conversation': [], 'custom_data': {}}

        session_data = st.session_state.conversations[session_id]

        with st.sidebar:
            st.title("Configuration:")
            chatbot_name = st.text_input("Chatbot Name")
            role_options = ["Customer Support", "Sales Assistant", "Technical Support", "HR Assistant"]
            role = st.selectbox("Chatbot Role and Objective", role_options)
            company_name = st.text_input("Company Name")
            activity_domain_options = ["Tourism", "Healthcare", "Transport", "Telecom", "Sport", "Finance"]
            activity_domain = st.selectbox("Activity Domain", activity_domain_options)
            instructions_options = ["Provide detailed answers", "Be concise", "Use friendly language", "Be formal"]
            instructions = st.selectbox("Instructions for the Chatbot", instructions_options)
            phone_number = st.text_input("Phone Number")
            social_media = st.text_input("Social Media (e.g., Twitter, LinkedIn)")

            st.title("Menu:")
            self.uploaded_files = st.file_uploader(
                "Upload your Files and Click on the Submit & Process Button",
                accept_multiple_files=True,
                type=['pdf', 'csv', 'txt', 'xls', 'json']
            )
            self.url = st.text_input("Enter a URL to process")
            self.custom_data_input = st.text_area("Enter custom data (e.g., 'key1: value1\nkey2: value2')")

            if st.button("Submit & Process"):
                if not all([chatbot_name, role, company_name, activity_domain, instructions]):
                    st.error("Please fill in all the configuration settings before submitting.")
                    return
                
                # Initialisation du chatbot avec les paramètres configurés
                self.chatbot = Chatbot(
                    chatbot_name=chatbot_name,
                    role=role,
                    company_name=company_name,
                    activity_domain=activity_domain,
                    instructions=instructions,
                    phone_number=phone_number,
                    social_media=social_media
                )
                st.session_state.chatbot = self.chatbot
                st.success("Chatbot has been initialized successfully.")
                self.process_files_and_url()

        user_question = st.text_input("Ask a Question")
        if user_question:
            self.handle_user_input(user_question, session_id)

        self.display_conversation(session_id)

    def process_files_and_url(self):
        if self.chatbot is None:
            st.error("Chatbot is not initialized. Please fill in the configuration settings and submit.")
            return
        
        with st.spinner("Processing..."):
            try:
                vector_store_folder = self.chatbot.process_files_and_url(self.uploaded_files, self.url, st.session_state.session_id)
                st.session_state.conversations[st.session_state.session_id]['vector_store_folder'] = vector_store_folder

                if self.custom_data_input:
                    custom_data_entries = self.custom_data_input.split('\n')
                    custom_data_dict = {}
                    for entry in custom_data_entries:
                        if ':' in entry:
                            key, value = entry.split(':', 1)
                            custom_data_dict[key.strip()] = value.strip()
                    st.session_state.conversations[st.session_state.session_id]['custom_data'] = custom_data_dict

                st.success("Processing completed successfully!")
            except ValueError as e:
                st.error(str(e))

    def handle_user_input(self, user_question, session_id):
        if 'vector_store_folder' not in st.session_state.conversations.get(session_id, {}):
            st.error("Please upload and process files or URLs first.")
            return

        if self.chatbot is None:
            st.error("Chatbot is not initialized. Please fill in the configuration settings and submit.")
            return

        answer = self.chatbot.handle_user_input(user_question, session_id)
        st.session_state.conversations[session_id]['conversation'].append({"question": user_question, "answer": answer})

    def display_conversation(self, session_id):
        st.subheader("Conversation History")
        conversation = st.session_state.conversations.get(session_id, {}).get('conversation', [])
        for i, convo in enumerate(conversation):
            cols = st.columns(2)
            with cols[0]:
                st.write(f"**Question {i+1}:** {convo['question']}")
            with cols[1]:
                st.write(f"**Answer {i+1}:** {convo['answer']}")

if __name__ == "__main__":
    ChatbotApp()
