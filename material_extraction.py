import re
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Path to the folder containing PDF files
folder_path = "data"

# Initialize an empty list to hold all extracted and processed text chunks
all_chunks = []

# -----------------------------
# Function: clean_pdf_text
# -----------------------------
def clean_pdf_text(text: str) -> str:
    """
    Cleans up raw text extracted from PDFs:
    - Removes lines like "CHAPTER 1"
    - Removes standalone page numbers
    - Strips whitespace and removes empty lines
    """
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        if re.match(r'^\s*chapter\s+\d+', line, re.IGNORECASE):
            continue  # Skip "CHAPTER X" headers
        if re.match(r'^\d+$', line):
            continue  # Skip standalone numbers (page numbers)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

# -----------------------------
# Function: split_into_sections
# -----------------------------
def split_into_sections(text: str):
    """
    Splits cleaned text into sections using numbered headers like "2.3 Some Title"
    Ignores duplicate headers if very little content follows them (likely an OCR artifact)
    """
    header_pattern = re.compile(r'^\d+\.\d+\s+.+$', re.MULTILINE)

    lines = text.splitlines()
    sections = []
    current_header = None
    buffer = []

    for line in lines:
        if header_pattern.match(line):
            header = line.strip()
            # Skip repeated headers if the previous buffer is too short
            if header == current_header and len(" ".join(buffer).strip()) < 50:
                buffer = []
                continue
            # Save the previous section before starting a new one
            if current_header:
                sections.append((current_header, "\n".join(buffer)))
                buffer = []
            current_header = header
        else:
            buffer.append(line)

    # Append the last buffered section
    if current_header:
        sections.append((current_header, "\n".join(buffer)))

    return sections

# -----------------------------
# Setup text splitter
# -----------------------------
# Breaks long sections into smaller overlapping chunks (to preserve context)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)

# -----------------------------
# Function: deduplicate_results
# -----------------------------
def deduplicate_results(results):
    """Removes duplicate search results based on document content."""
    seen = set()
    unique_results = []
    for doc, score in results:
        content = doc.page_content
        if content not in seen:
            seen.add(content)
            unique_results.append((doc, score))
    return unique_results

# -----------------------------
# Extract and process PDFs
# -----------------------------
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(folder_path, filename)
        reader = PdfReader(pdf_path)

        # Concatenate all text from PDF pages
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        # Clean up text
        cleaned = clean_pdf_text(text)

        # Split into logical sections based on headers
        sections = split_into_sections(cleaned)

        # Chunk sections if too long; otherwise store as-is
        for header, content in sections:
            if len(content) < 1000:
                all_chunks.append({
                    "source": filename,
                    "section": header,
                    "text": content
                })
            else:
                # Use text splitter for long content
                sub_chunks = splitter.split_text(content)
                for i, chunk in enumerate(sub_chunks):
                    all_chunks.append({
                        "source": filename,
                        "section": header,
                        "chunk_index": i,
                        "text": chunk
                    })

# -----------------------------
# Convert to LangChain Documents
# -----------------------------
documents = []
for chunk in all_chunks:
    # Add metadata like source filename and section header
    metadata = {
        "source": chunk["source"],
        "section": chunk.get("section", ""),
    }
    if "chunk_index" in chunk:
        metadata["chunk_index"] = chunk["chunk_index"]

    # Create a LangChain Document object
    documents.append(
        Document(page_content=chunk["text"], metadata=metadata)
    )

print(f"Total Documents: {len(documents)}")

# -----------------------------
# Create embeddings and Chroma vector store
# -----------------------------
# Embedding model to convert text into numerical vectors
embedding_function = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# Specify where to store the vector index
persist_directory = "chroma_db"

# Create and persist the vector store
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_function,
    persist_directory=persist_directory
)

print(f"Chroma vector store created with {len(documents)} documents.")

# -----------------------------
# Retrieval-Augmented Generation Setup
# -----------------------------
# Create a retriever to fetch top-k relevant documents
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Load OpenAI Chat model (GPT-3.5-turbo)
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)

# Create a QA chain that uses the retriever + LLM to answer questions
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# -----------------------------
# Interactive Query Loop
# -----------------------------
# Keep asking user for input until they type "exit"
while True:
    user_query = input("\nAsk a question (or type 'exit'): ")
    if user_query.lower() == "exit":
        break
    response = qa_chain.run(user_query)
    print("\nAnswer:\n", response)