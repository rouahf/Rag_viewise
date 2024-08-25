import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
import json
import os
import uuid
from langchain.document_loaders import UnstructuredURLLoader

class VectorStoreManager:
    def __init__(self):
        self.vector_store_folder = None
        self.questions_file = "questions.json"

    def create_vector_store(self, text_chunks, index_name):
        if not text_chunks:
            raise ValueError("Les text_chunks sont vides. Vérifiez que les fichiers et URL ont été traités correctement.")
        
        self.vector_store_folder = f"faiss_index_{index_name}"
        os.makedirs(self.vector_store_folder, exist_ok=True)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embedding_values = embeddings.embed_documents(text_chunks)
        
        if not embedding_values:
            raise ValueError("Les embeddings sont vides. Le modèle n'a pas pu générer d'embeddings.")
        
        vector_store = FAISS.from_texts(text_chunks, embedding=embedding_values)
        vector_store.save_local(self.vector_store_folder)
        self.save_questions([])  # Initialiser avec une liste vide de questions
        return self.vector_store_folder

    # Les autres méthodes de la classe



    def load_vector_store(self, folder):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)

    def save_questions(self, questions):
        questions_path = os.path.join(self.vector_store_folder, self.questions_file)
        with open(questions_path, "w") as f:
            json.dump(questions, f)

    def load_questions(self, folder):
        questions_path = os.path.join(folder, self.questions_file)
        if os.path.exists(questions_path):
            with open(questions_path, "r") as f:
                return json.load(f)
        return []

class EmbeddingManager:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        # Set up API configuration if necessary
        # For GoogleGenerativeAIEmbeddings, it might not need explicit configuration here

    def process_files_and_url(self, uploaded_files, url):
        raw_text = self.get_all_text_from_files(uploaded_files)
        if url:
            raw_text += self.get_url_text(url)
        text_chunks = self.get_text_chunks(raw_text)
        return text_chunks

    def get_all_text_from_files(self, uploaded_files):
        raw_text = ""
        for file in uploaded_files:
            if file.name.endswith('.pdf'):
                raw_text += self.get_pdf_text(file)
            elif file.name.endswith('.csv'):
                raw_text += self.get_csv_text(file)
            elif file.name.endswith('.txt'):
                raw_text += self.get_txt_text(file)
            elif file.name.endswith('.xls'):
                raw_text += self.get_xls_text(file)
            elif file.name.endswith('.json'):
                raw_text += self.get_json_text(file)
        return raw_text

    def get_pdf_text(self, pdf):
        text = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def get_csv_text(self, csv):
        df = pd.read_csv(csv)
        return df.to_string(index=False)

    def get_txt_text(self, txt):
        return txt.read().decode("utf-8")

    def get_xls_text(self, xls):
        df = pd.read_excel(xls)
        return df.to_string(index=False)

    def get_json_text(self, json_file):
        data = json.load(json_file)
        return json.dumps(data, indent=2)

    def get_url_text(self, url):
        loader = UnstructuredURLLoader(urls=[url])
        documents = loader.load()
        return "\n".join([doc.page_content for doc in documents])

    def get_text_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return text_splitter.split_text(text)


class ChatbotApp:
    def __init__(self):
        self.vector_store_manager = VectorStoreManager()
        self.embedding_manager = EmbeddingManager()
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
            st.session_state.conversations[session_id] = {
                'conversation': [],
                'custom_data': {},
                'questions': [],
                'faiss_index': None
            }

        session_data = st.session_state.conversations[session_id]

        with st.sidebar:
            st.title("Configuration:")
            self.chatbot_name = st.text_input("Chatbot Name")
            role_options = ["Customer Support", "Sales Assistant", "Technical Support", "HR Assistant"]
            self.role = st.selectbox("Chatbot Role and Objective", role_options)
            self.company_name = st.text_input("Company Name")
            activity_domain_options = ["Tourism", "Healthcare", "Transport", "Telecom", "Sport", "Finance"]
            self.activity_domain = st.selectbox("Activity Domain", activity_domain_options)
            instructions_options = ["Provide detailed answers", "Be concise", "Use friendly language", "Be formal"]
            self.instructions = st.selectbox("Instructions for the Chatbot", instructions_options)
            self.phone_number = st.text_input("Phone Number")
            self.social_media = st.text_input("Social Media (e.g., Twitter, LinkedIn)")
            self.faiss_index_name = st.text_input("Enter a name for the FAISS index")

            st.title("Menu:")
            self.uploaded_files = st.file_uploader(
                "Upload your Files and Click on the Submit & Process Button",
                accept_multiple_files=True,
                type=['pdf', 'csv', 'txt', 'xls', 'json']
            )
            self.url = st.text_input("Enter a URL to process")
            self.custom_data_input = st.text_area("Enter custom data (e.g., 'key1: value1\nkey2: value2')")

            if st.button("Submit & Process"):
                self.process_files_and_url()

        self.display_previous_questions(session_id)
        user_question = st.text_input("Ask a Question")
        if user_question:
            self.handle_user_input(user_question, session_id)

        self.display_conversation(session_id)

    def display_previous_questions(self, session_id):
        st.subheader("Previously Asked Questions")
        questions = st.session_state.conversations[session_id]['questions']
        if questions:
            for i, question in enumerate(questions):
                st.write(f"**Question {i+1}:** {question}")
        else:
            st.write("No previous questions found.")

    def process_files_and_url(self):
        session_id = st.session_state.session_id
        text_chunks = self.embedding_manager.process_files_and_url(self.uploaded_files, self.url)
       
        if not self.faiss_index_name:
            st.error("Please enter a name for the FAISS index.")
            return

        folder = self.vector_store_manager.create_vector_store(text_chunks, self.faiss_index_name)
    
        st.session_state.conversations[session_id]['faiss_index'] = folder
        st.session_state.conversations[session_id]['questions'] = []

        if self.custom_data_input:
            custom_data = dict(line.split(':') for line in self.custom_data_input.splitlines())
            st.session_state.conversations[session_id]['custom_data'] = custom_data

    def get_conversational_chain(self):
        prompt_template = """
        You are {chatbot_name}, a chatbot that assists with {role} for {company_name}. Your role involves providing support in the field of {activity_domain}.
        Your instructions are: {instructions}.
        If a user inquires about the company, your phone number is {phone_number} and your social media handle is {social_media}.
        Given these details and the documents provided, answer the user's questions accurately.
        """
        
        # Create the Chat model
        model_name = "models/embedding-001"  # Ensure this is the correct model name
        chat_model = ChatGoogleGenerativeAI(model=model_name)

        # Create a PromptTemplate
        template = PromptTemplate(template=prompt_template)
        
        # Load QA chain with chat model and template
        return load_qa_chain(chat_model, prompt_template=template)

    def handle_user_input(self, user_question, session_id):
        faiss_index_folder = st.session_state.conversations[session_id].get('faiss_index')
        if not faiss_index_folder:
            st.error("Please upload and process files or URLs first.")
            return

        try:
            new_db = self.vector_store_manager.load_vector_store(faiss_index_folder)
            docs = new_db.similarity_search(user_question)
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return

        custom_data = st.session_state.conversations[session_id]['custom_data']
        answer = custom_data.get(user_question, None)

        if not answer:
            chain = self.get_conversational_chain()
            answer = chain.run(input_documents=docs, question=user_question)

        st.session_state.conversations[session_id]['conversation'].append({
            'question': user_question,
            'answer': answer
        })

        st.write(f"**Question:** {user_question}")
        st.write(f"**Answer:** {answer}")

    def display_conversation(self, session_id):
        st.subheader("Conversation")
        conversation = st.session_state.conversations[session_id]['conversation']
        for entry in conversation:
            st.write(f"**Question:** {entry['question']}")
            st.write(f"**Answer:** {entry['answer']}")

if __name__ == "__main__":
    ChatbotApp()