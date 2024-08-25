import os
import json
import pandas as pd
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
import google.generativeai as genai

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
