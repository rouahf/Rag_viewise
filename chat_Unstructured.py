import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
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
        self.id_file = "last_index_id.json"

    def _get_next_id(self):
        # Lire le dernier ID utilisé depuis le fichier
        if os.path.exists(self.id_file):
            with open(self.id_file, "r") as f:
                data = json.load(f)
                last_id = data.get("last_id", 0)
        else:
            last_id = 0

        # Déterminer le prochain ID et mettre à jour le fichier
        next_id = last_id + 1
        with open(self.id_file, "w") as f:
            json.dump({"last_id": next_id}, f)

        return next_id

    def create_vector_store(self, text_chunks):
        next_id = self._get_next_id()
        self.vector_store_folder = f"faiss_index_{next_id}"
        os.makedirs(self.vector_store_folder, exist_ok=True)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(self.vector_store_folder)
        return self.vector_store_folder

    def load_vector_store(self, folder):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)

class EmbeddingManager:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)

    def process_files_and_url(self, uploaded_files, url):
        raw_text = self.get_all_text_from_files(uploaded_files)
        if url:
            raw_text += self.get_url_text(url)
        text_chunks = self.get_text_chunks(raw_text)
        return text_chunks

    def get_all_text_from_files(self, uploaded_files):
        raw_text = ""
        pdf_docs = [file for file in uploaded_files if file.name.endswith('.pdf')]
        csv_docs = [file for file in uploaded_files if file.name.endswith('.csv')]
        txt_docs = [file for file in uploaded_files if file.name.endswith('.txt')]
        xls_docs = [file for file in uploaded_files if file.name.endswith('.xls')]
        json_docs = [file for file in uploaded_files if file.name.endswith('.json')]

        if pdf_docs:
            raw_text += self.get_pdf_text(pdf_docs)
        if csv_docs:
            raw_text += self.get_csv_text(csv_docs)
        if txt_docs:
            raw_text += self.get_txt_text(txt_docs)
        if xls_docs:
            raw_text += self.get_xls_text(xls_docs)
        if json_docs:
            raw_text += self.get_json_text(json_docs)

        return raw_text

    def get_pdf_text(self, pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def get_csv_text(self, csv_docs):
        text = ""
        for csv in csv_docs:
            df = pd.read_csv(csv)
            text += df.to_string(index=False)
        return text

    def get_txt_text(self, txt_docs):
        text = ""
        for txt in txt_docs:
            text += txt.read().decode("utf-8")
        return text

    def get_xls_text(self, xls_docs):
        text = ""
        for xls in xls_docs:
            df = pd.read_excel(xls)
            text += df.to_string(index=False)
        return text

    def get_json_text(self, json_docs):
        text = ""
        for json_file in json_docs:
            data = json.load(json_file)
            text += json.dumps(data, indent=2)
        return text

    def get_url_text(self, url):
        loader = UnstructuredURLLoader(urls=[url])
        documents = loader.load()
        text = "\n".join([doc.page_content for doc in documents])
        return text

    def get_text_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks

class ChatbotApp:
    def __init__(self):
        self.vector_store_manager = VectorStoreManager()
        self.embedding_manager = EmbeddingManager()
        self.setup_streamlit()

    def setup_streamlit(self):
        st.set_page_config("Chat with Files")
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

        user_question = st.text_input("Ask a Question")
        if user_question:
            self.handle_user_input(user_question, session_id)

        self.display_conversation(session_id)

    def process_files_and_url(self):
        with st.spinner("Processing..."):
            text_chunks = self.embedding_manager.process_files_and_url(self.uploaded_files, self.url)
            if text_chunks:
                vector_store_folder = self.vector_store_manager.create_vector_store(text_chunks)
                session_id = st.session_state.session_id
                st.session_state.conversations[session_id]['vector_store_folder'] = vector_store_folder

                # Process and store custom data as general context
                if self.custom_data_input:
                    custom_data_entries = self.custom_data_input.split('\n')
                    custom_data_dict = {}
                    for entry in custom_data_entries:
                        if ':' in entry:
                            key, value = entry.split(':', 1)
                            custom_data_dict[key.strip()] = value.strip()
                    st.session_state.conversations[session_id]['custom_data'] = custom_data_dict

                st.success("Processing completed successfully!")
            else:
                st.error("No valid text found in the provided files and URL.")

    def get_conversational_chain(self, chatbot_name, role, company_name, activity_domain, instructions, phone_number, social_media):
        prompt_template = f"""
        Your name is {chatbot_name}, a chatbot for {company_name} operating in the {activity_domain} domain.
        Your role is {role}.
        Here are your instructions: {instructions}.
        Your contact phone number is {phone_number}.
        Your social media profile is {social_media}.
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
        if 'vector_store_folder' not in st.session_state.conversations.get(session_id, {}):
            st.error("Please upload and process files or URLs first.")
            return

        vector_store_folder = st.session_state.conversations[session_id]['vector_store_folder']
        new_db = self.vector_store_manager.load_vector_store(vector_store_folder)
        docs = new_db.similarity_search(user_question)

        custom_data = st.session_state.conversations[session_id]['custom_data'] if st.session_state.conversations[session_id]['custom_data'] else {}

        # First check custom data for a direct answer
        answer = custom_data.get(user_question, None)
        if not answer:
            # If no direct answer, use the QA chain
            chain = self.get_conversational_chain(
                self.chatbot_name, self.role, self.company_name, self.activity_domain, self.instructions, self.phone_number, self.social_media
            )
            response = chain({"input_documents": docs, "question": user_question, "custom_data": json.dumps(custom_data)})
            answer = response.get("output_text", "Je n'ai pas la réponse à cette question.")

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
