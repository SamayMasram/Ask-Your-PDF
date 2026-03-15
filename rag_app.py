# -*- coding: utf-8 -*-

from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchRun


def format_docs(docs: List[Document]):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def load_and_split_pdfs(pdf_paths):

    docs = []

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(docs)


# ---------------- PDF RAG ---------------- #

def build_rag_chain(pdf_paths, api_key):

    chunks = load_and_split_pdfs(pdf_paths)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=api_key
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=api_key
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant.

Use ONLY the provided PDF context to answer the question.

If the answer is not in the document, say:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}

Answer clearly.
"""
    )

    def rag_pipeline(question: str):

        # Newer LangChain retrievers use .invoke instead of .get_relevant_documents
        docs = retriever.invoke(question)

        context = format_docs(docs)

        response = llm.invoke(
            prompt.format(context=context, question=question)
        )

        return {
            "answer": response.content,
            "sources": docs
        }

    return rag_pipeline


# ---------------- PDF + WEB ---------------- #

def build_pdf_web_chain(pdf_paths, api_key):

    chunks = load_and_split_pdfs(pdf_paths)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=api_key
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    web_search = DuckDuckGoSearchRun()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )

    prompt = ChatPromptTemplate.from_template(
        """
Use BOTH PDF context and web search results to answer.

Prefer the PDF for document-specific details.

PDF Context:
{pdf_context}

Web Context:
{web_context}

Question:
{question}

Answer clearly and structured.
"""
    )

    def rag_web_pipeline(question: str):

        # Newer LangChain retrievers use .invoke instead of .get_relevant_documents
        pdf_docs = retriever.invoke(question)

        pdf_context = format_docs(pdf_docs)

        web_context = web_search.run(question)

        response = llm.invoke(
            prompt.format(
                pdf_context=pdf_context,
                web_context=web_context,
                question=question
            )
        )

        return {
            "answer": response.content,
            "sources": pdf_docs
        }

    return rag_web_pipeline
