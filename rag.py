import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

# Correct Supabase pgvector imports
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector as PGVectorStore

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

COLLECTION_NAME = "pdf_rag"
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env – check Supabase connection string")

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

    loader = PyMuPDFLoader(file_path)
    raw_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    docs = text_splitter.split_documents(raw_docs)

    try:
        PGVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            connection=DATABASE_URL,
            collection_name=COLLECTION_NAME,
            use_jsonb=True,
        )
        print(f"✅ Ingested {len(docs)} chunks into Supabase pgvector")
    except Exception as e:
        print(f"Ingestion error: {str(e)}")
        raise

def get_rag_chain():
    embeddings = get_embeddings()

    try:
        print("Loading vector store from Supabase...")
        vector_store = PGVectorStore(
            connection=DATABASE_URL,
            embeddings=embeddings,          # ← CORRECT parameter name here
            collection_name=COLLECTION_NAME,
        )
        print("Vector store loaded successfully")

        retriever = vector_store.as_retriever(search_kwargs={"k": 6})

        llm = get_llm()

        prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context.
If the answer is not in the context, say "I don't have enough information".

Context:
{context}

Question: {input}
Answer:"""
        )

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever, question_answer_chain)

    except Exception as e:
        print(f"Failed to load vector store: {str(e)}")
        raise RuntimeError(f"Failed to initialize RAG chain: {str(e)}\nCheck DATABASE_URL and Supabase connection.")

def query_document(question: str) -> str:
    try:
        chain = get_rag_chain()
        result = chain.invoke({"input": question})
        return result["answer"].strip()
    except Exception as e:
        return f"Error during query: {str(e)}\nMake sure a PDF was uploaded and Supabase is reachable."