import os
import time
import base64
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

load_dotenv()

app = FastAPI(title="Mulimodal RAG BACkEND WITH RERANKER")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./db/chroma_db", embedding_function=embeddings)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


class QueryRequest(BaseModel):
    question: str

def call_gemini_with_retry(contents, model_name="gemini-3.8-flash", max_retries=4):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents = contents
                )
            return response.text.strip()

        except Exception as e:
            err = str(e)
            if ("503" in err or "429" in err) and attempt < max_retries- 1:
                time.sleep(2**(attempt + 1))
            else:
                raise e

@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    initial_docs = db.similarity_search(request.question, k=10)
    if not initial_docs:
        return {"answer": "No relevant documents found in the database.", "sources": []}
    pairs = [[request.question, doc.page_content] for doc in initial_docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored_docs[:3]]
    multimodal_contents = [
    "Answer the user's question strictly using the provided document text, HTML tables, and images.\n\n"
    ]
    sources_used = []
    for i, doc in enumerate(top_docs):
        multimodal_contents.append(f"--- Document {i + 1} ---\n{doc.page_content}\n")
        raw_table = doc.metadata.get("raw_table_html", "")
        if raw_table:
            multimodal_contents.append(f"[Document {i + 1} exact HTML Table]:\n{raw_table}\n")
        raw_image_b64 = doc.metadata.get("raw_image_b64", "")
        if raw_image_b64:
            image_bytes = base64.b64decode(raw_image_b64)
            mime_type = "image/png" if image_bytes.startswith(b"\x89PNG") else "image/jpeg"
            multimodal_contents.append(
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
        sources_used.append({
            "source": os.path.basename(doc.metadata.get("source", "unknown")),
            "chunk_index": doc.metadata.get("chunk_index", -1),
            "has_table": doc.metadata.get("has_table", False),
            "has_image": doc.metadata.get("has_image", False),
            "raw_table_html": raw_table,
            "raw_image_b64": raw_image_b64
            })
    multimodal_contents.append(f"\nUser Question: {request.question}")
    answer = call_gemini_with_retry(multimodal_contents)
    return {"answer": answer, "sources": sources_used}