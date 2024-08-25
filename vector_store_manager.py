import os
import json
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class VectorStoreManager:
    def __init__(self):
        self.vector_store_folder = None
        self.id_file = "last_index_id.json"
        load_dotenv()  # Load environment variables from a .env file

        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    def _get_next_id(self):
        if os.path.exists(self.id_file):
            with open(self.id_file, "r") as f:
                data = json.load(f)
                last_id = data.get("last_id", 0)
        else:
            last_id = 0

        next_id = last_id + 1
        with open(self.id_file, "w") as f:
            json.dump({"last_id": next_id}, f)

        return next_id

    def create_vector_store(self, text_chunks, index_name):
        if not text_chunks:
            raise ValueError("Les text_chunks sont vides. Vérifiez que les fichiers et URL ont été traités correctement.")
        
        self.vector_store_folder = f"faiss_index_{index_name}"
        os.makedirs(self.vector_store_folder, exist_ok=True)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.google_api_key)
        embedding_values = embeddings.embed_documents(text_chunks)
        
        if not embedding_values:
            raise ValueError("Les embeddings sont vides. Le modèle n'a pas pu générer d'embeddings.")
        
        vector_store = FAISS.from_texts(text_chunks, embeddings)
        vector_store.save_local(self.vector_store_folder)
        self.save_questions([])  # Initialiser avec une liste vide de questions
        return self.vector_store_folder

    def load_vector_store(self, id):
        folder = f"faiss_index_{id}"
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.google_api_key)
        vector_store = FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)
        print(folder)
        return vector_store

    def save_questions(self, questions):
        with open("questions.json", "w") as f:
            json.dump({"questions": questions}, f)

def main():
    manager = VectorStoreManager()


    # Load the vector store
    vector_store = manager.load_vector_store(1)
    print(f"Loaded vector store from folder: {1}")

if __name__ == "__main__":
    main()
