import json
import streamlit as st
from vector_store_manager import VectorStoreManager
from embedding_manager import EmbeddingManager
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

class Chatbot:
    def __init__(self, chatbot_name="", role="", company_name="", activity_domain="", instructions="", phone_number="", social_media=""):
        self.chatbot_name = chatbot_name
        self.role = role
        self.company_name = company_name
        self.activity_domain = activity_domain
        self.instructions = instructions
        self.phone_number = phone_number
        self.social_media = social_media
        self.vector_store_manager = VectorStoreManager()
        self.embedding_manager = EmbeddingManager()

    def process_files_and_url(self, uploaded_files, url, session_id):
        text_chunks = self.embedding_manager.process_files_and_url(uploaded_files, url)
        if text_chunks:
            vector_store_id = self.vector_store_manager._get_next_id()  # Generate a new ID
            vector_store_folder = self.vector_store_manager.create_vector_store(text_chunks, vector_store_id)
            st.session_state.conversations[session_id]['vector_store_id'] = vector_store_id  # Save vector store ID in session
            return vector_store_id  # Return the ID instead of folder name
        else:
            raise ValueError("No valid text found in the provided files and URL.")

    def get_conversational_chain(self):
        prompt_template = f"""
        Your name is {self.chatbot_name}, a chatbot for {self.company_name} operating in the {self.activity_domain} domain.
        Your role is {self.role}.
        Here are your instructions: {self.instructions}.
        Your contact phone number is {self.phone_number}.
        Your social media profile is {self.social_media}.
        Answer the question based on the provided context and the custom data provided.
        
        Context:\n {{context}}\n
        Custom Data:\n {{custom_data}}\n
        Question: \n{{question}}\n

        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "custom_data", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

    def handle_user_input(self, user_question, session_id):
        vector_store_id = st.session_state.conversations[session_id].get('vector_store_id')
        if not vector_store_id:
            raise ValueError("Vector store ID is not set. Please process files and URL first.")
        
        vector_store = self.vector_store_manager.load_vector_store(vector_store_id)
        docs = vector_store.similarity_search(user_question)
        custom_data = st.session_state.conversations[session_id].get('custom_data', {})

        # Check custom data first
        answer = custom_data.get(user_question, None)
        if not answer:
            chain = self.get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question, "custom_data": json.dumps(custom_data)})
            answer = response.get("output_text", "Je n'ai pas la réponse à cette question.")
        
        return answer
