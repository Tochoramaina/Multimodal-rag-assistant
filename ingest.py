import os
import shutil
import base64
import time
import io 
from concurrent.futures import ThreadPoolExecutor 
from PIL import Image
from dotenv import load_dotenv
from  google import genai
from google.genai import types 
from unstructured.partition.pdf import partition_pdf 
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
#from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()
DATA_PATH = "./data/"
PERSIST_DIRECTORY = "./db/chroma_db"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MIN_IMAGE_DIMENSION =150

def is_valid_image(image_b64:str) -> bool:
    if not image_b64:
        return False
    try:
        image_bytes = base64.b64decode(image_b64) 
        with Image.open(io.BytesIO(img_bytes)) as img:
            width, height = img.size 
            return width >= MIN_IMAGE_DIMENSION and height >= MIN_IMAGE_DIMENSION
    except Exception:
        return  False
def summarize_multimodal_chunk(text:str, table_html:str =None, image_b64:str=None, max_retries:int=4) -> str:
    contents = [
        "Summarize this document element concisely for semantic search retrieval. "
        "Preserve all exact figures, column names, labels and relationships.\n\n"
    ]
    if text:
        contents.append(f"Surrounding Text:\n{text}\n")
    if table_html:
        contents.append(f"HTML Table:\n{table_html}\n")
    if image_b64:
        contents.append(
            types.Part.from_bytes(data=base64.b64decode(image_b64),
                mime_type="image/jpeg")
            )
    for attempt in range(max_retries):
        time.sleep(1.5)
        try:
            response = client.models.generate_content(
                model ="gemini-3.8-flash",
                contents =contents)
            return response.text.strip()
        except Exception as e:
            err_msg = str(e)
            if("429" in err_msg or "503" in err_msg) and attempt < max_retries -1:
                wait_time = 2 **(attempt + 1)
                print(f"Rate limit/transient error, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

def process_single_chunk(args:tuple[int, int, object, str]) -> Document:
    idx, total, chunk, filepath = args 
    orig_elements = getattr(chunk.metadata, "orig_elements", [])
    tables = [el.metadata.text_as_html for el in orig_elements if el.category =="Table" and getattr(el.metadata, "text_as_html", None)]
    images = [el.metadata.image_base64 for el in orig_elements if el.category =="Image" and is_valid_image(getattr(el.metadata, "image_base64", None))]
    if tables or images:
        print(f"-----> Summarizing Multimodal chunk {idx + 1}/{total} ...")
        summary = summarize_multimodal_chunk(
            text = chunk.text,
            table_html = tables[0] if tables else None,
            image_b64 = images[0] if images else None 
            )
        page_content = f"{chunk.text}\n\n[Visual/Table Summary]: {summary}"
    else:
        page_content = chunk.text 
    metadata = {
       "source": filepath,
       "chunk_idex": idx,
       "has_table": bool(tables),
       "has_image": bool(images),
       "raw_table_html": tables[0] if tables else "",
       "raw_image_b64": images[0] if images else ""
    }
    return Document(page_content=page_content, metadata=metadata)

def process_pdf_to_documents(filepath:str) -> list[Document]:
    print(f"Partitioning {filepath} into atomic elements ...")
    start_time = time.time()
    raw_elements = partition_pdf(filename=filepath,
        strategy="hi_res", infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True)
    print(f"Grouping atomic elements by title finished in {time.time() - start_time:.1f}")
    title_chunks = chunk_by_title(raw_elements, max_characters=2000,
        new_after_n_chars=1500, combine_text_under_n_chars=300)

    total = len(title_chunks)
    tasks = [(idx, total, chunk, filepath) for idx, chunk in enumerate(title_chunks)]
    with ThreadPoolExecutor(max_workers =2) as executor:
        langchain_docs = list(executor.map(process_single_chunk, tasks))
    return langchain_docs


def build_vector_store():
    pdf_files = [
    os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF found in {DATA_PATH}. Add your .pdf files there.")
    all_documents = []  
    for pdf_path in pdf_files:
        all_documents.extend(process_pdf_to_documents(pdf_path))
    print(f"Created {len(all_documents)} total Multimodal chunks.")

    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)

    print("Generating embeddings and saving to Chroma DB...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    Chroma.from_documents(
        all_documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("Multimodal Vector database successfully built and stored locally!")

if __name__ == "__main__":
    build_vector_store()
