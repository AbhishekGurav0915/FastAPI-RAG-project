import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # ← new import

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

COLLECTION_NAME = "pdf_rag"
CHROMA_PATH = "./chroma_db"  # persistent folder on disk

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={"device": "cpu"},
    )

def get_llm():
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    elif os.getenv("GEMINI_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    raise ValueError("No LLM key in .env")

def ingest_document(file_path: str):
    embeddings = get_embeddings()

    # Load & chunk PDF
    loader = PyMuPDFLoader(file_path)
    raw_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    docs = text_splitter.split_documents(raw_docs)

    # Use Chroma – simple & persistent
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,  # saves to disk
        collection_name=COLLECTION_NAME,
    )
    print(f"✅ Ingested {len(docs)} chunks into Chroma")

def get_rag_chain():
    embeddings = get_embeddings()

    # Load existing Chroma DB from disk
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 6})

    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        """Answer based only on this context. If not enough info, say so.

Context:
{context}

Question: {input}
Answer:"""
    )

    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)

def query_document(question: str) -> str:
    try:
        chain = get_rag_chain()
        result = chain.invoke({"input": question})
        return result["answer"].strip()
    except Exception as e:
        return f"Error: {str(e)}\n\nUpload a PDF first?"