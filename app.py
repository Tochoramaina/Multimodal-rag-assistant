import os
import time
import base64
from google.genai import types
from google import genai
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import time
load_dotenv()


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PERSIST_DIRECTORY = "./db/chroma_db"
if not os.path.exists(PERSIST_DIRECTORY):
    raise FileNotFoundError("Vector database not found! Please run 'ingest.py' first.")

print("Loading local vector store...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
retriever = db.as_retriever(search_type='similarity', search_kwargs={"k": 3})

chat_history = []

def call_gemini_with_retry(prompt, model_name="gemini-3.8-flash", max_retries=3):
    for attempts in range(max_retries):
        try:
            response = client.models.generate_content(model=model_name,
                contents=prompt
                )
            return response.text.strip()
        except Exception as e:
            if "503" in str(e) and attempts < max_retries - 1:
                print(f"503 service unavailable, retring in 2 seconds (attemp {attempts})")
                time.sleep(2)
            else:
                raise e

def ask_question(user_question):
    global chat_history
    print(f'\n--- You asked: {user_question} ---')
    if chat_history:
        history_str = "\n".join([f"{msg['role']}: {msg['text']}" for msg in chat_history])
        rephrase_prompt = (
            f"Given the conversation history below, rewrite the user's latest question "
            f"into a standalone search query. Respond with ONLY the query string.\n\n"
            f"History:\n{history_str}\n\n"
            f"Latest Question: {user_question}"
        )
         
        search_query = call_gemini_with_retry(rephrase_prompt)
        print(f'Rephrased query: {search_query}')
    else:
        search_query = user_question

    
    relevant_docs = retriever.invoke(search_query)
    history_str = "\n".join([f"{msg['role']}: {msg['text']}" for msg in chat_history]) if chat_history else "None"
    multimodal_contents = [f"""You are a helpful assistant. Answer the user question based strictly on the provided context, tables, images and past chat history.

                    --- Past Chat History ---
                    {history_str}


                    --- Retrieved Document chunks---
                    
                    """]
    for i, doc in enumerate(relevant_docs):
        multimodal_contents.append(f"\n[Doc {i + 1} Text & Summary]:\n{doc.page_content}\n") 
        raw_table = doc.metadata.get("raw_table_html", "")  
        if raw_table:
            multimodal_contents.append(f"[Doc {i + 1} exact HTML Table]:\n{raw_table}\n")  
        raw_image_b64= doc.metadata.get("raw_image_b64", "") 
        if raw_image_b64:
            print(f" -> Attaching visual image from  doc {i + 1} to gemini prompt...")  
            image_bytes = base64.b64decode(raw_image_b64) 
            multimodal_contents.append(
                types.Part.from_bytes(data=image_bytes, mime_types="image/jpeg")
                ) 

    multimodal_contents.append(f"\n--- Current Question ---\n{user_question}")



    answer = call_gemini_with_retry(multimodal_contents)
    chat_history.append({"role": "User", "text": user_question})
    chat_history.append({"role": "Assistant", "text": answer})

    print(f'\n--- AI says ---\n{answer}')
    return answer

def start_chat():
    print('\nRAG Chat Initialized. Type "quit" to exit.')
    while True:
        question = input("\nYour question: ")
        if question.lower().strip() == 'quit':
            print("Goodbye!")
            break
        if question.strip():
            ask_question(question)

if __name__ == "__main__":
    start_chat()