import re
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

folder_path = "data"

# Flat list of all chunks across all documents
all_chunks = []

def clean_pdf_text(text: str) -> str:
    """
    Clean up PDF text:
    - Remove running headers like 'CHAPTER 1' or 'CHAPTER 2'
    - Remove page numbers (lines that only contain digits)
    - Normalize spaces
    """
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^\s*chapter\s+\d+', line, re.IGNORECASE):
            continue
        if re.match(r'^\d+$', line):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def split_into_sections(text: str):
    """
    Split text into sections based on numeric headers.
    Handles repeated headers by ignoring duplicates if there's little content in between.
    """
    header_pattern = re.compile(r'^\d+\.\d+\s+.+$', re.MULTILINE)

    lines = text.splitlines()
    sections = []
    current_header = None
    buffer = []

    for line in lines:
        if header_pattern.match(line):
            header = line.strip()
            if header == current_header and len(" ".join(buffer).strip()) < 50:
                buffer = []
                continue
            if current_header:
                sections.append((current_header, "\n".join(buffer)))
                buffer = []
            current_header = header
        else:
            buffer.append(line)

    if current_header:
        sections.append((current_header, "\n".join(buffer)))

    return sections

# Create text splitter for chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)

def deduplicate_results(results):
    """Deduplicate search results based on page_content."""
    seen = set()
    unique_results = []
    for doc, score in results:
        content = doc.page_content
        if content not in seen:
            seen.add(content)
            unique_results.append((doc, score))
    return unique_results

# Process all PDFs
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(folder_path, filename)
        reader = PdfReader(pdf_path)

        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        cleaned = clean_pdf_text(text)
        sections = split_into_sections(cleaned)

        for header, content in sections:
            if len(content) < 1000:
                all_chunks.append({
                    "source": filename,
                    "section": header,
                    "text": content
                })
            else:
                sub_chunks = splitter.split_text(content)
                for i, chunk in enumerate(sub_chunks):
                    all_chunks.append({
                        "source": filename,
                        "section": header,
                        "chunk_index": i,
                        "text": chunk
                    })

# Convert chunks into LangChain Document objects
documents = []
for chunk in all_chunks:
    metadata = {
        "source": chunk["source"],
        "section": chunk.get("section", ""),
    }
    if "chunk_index" in chunk:
        metadata["chunk_index"] = chunk["chunk_index"]

    documents.append(
        Document(page_content=chunk["text"], metadata=metadata)
    )

print(f"Total Documents: {len(documents)}")

# Create OpenAI embeddings (PAID)
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
persist_directory = "chroma_db"

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_function,
    persist_directory=persist_directory
)

print(f"Chroma vector store created with {len(documents)} documents.")

# ---------------------------------------------------
# RetrievalQA: OpenAI LLM (GPT‑3.5‑turbo)
# ---------------------------------------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# GPT‑3.5‑turbo is fast and cheap
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# Example queries
while True:
    user_query = input("\nAsk a question (or type 'exit'): ")
    if user_query.lower() == "exit":
        break
    response = qa_chain.run(user_query)
    print("\nAnswer:\n", response)
