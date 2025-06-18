from flask import Flask, jsonify, request
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv


from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv("/Users/siddharthakosti/Downloads/catalyst_ragaai/RagaAI-Catalyst/.env", override=True)

app = Flask(__name__)

catalyst = RagaAICatalyst(
        access_key="ZlOlOngSZ6HInl9rr21d",
        secret_key="VQ5ybBrhSGoWIvVHaVQBkiGWM4DzRTOI96IQjSCI",
        base_url="http://4.224.253.24/api"
    )


@app.route('/mini', methods=['POST'])
def chat_with_mini():
    """
    Endpoint to chat with GPT-4o-mini
    Expects JSON with 'message' field
    """
    try:
        tracer = Tracer(
            project_name="multi_endpoint_rag_tracer",
            dataset_name="gpt_mini_v1",
            tracer_type="langchain",
            )

        def create_rag_pipeline(pdf_path):
            # 1. Load the PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # 2. Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(documents)
            
            # 3. Create embeddings
            embeddings = OpenAIEmbeddings()
            
            # 4. Create vector store
            vectorstore = FAISS.from_documents(texts, embeddings)
            
            # 5. Create retrieval QA chain
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            qa_chain = RetrievalQA.from_chain_type(
                llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
            )
            
            return qa_chain

        def main():
            # Path to your PDF
            pdf_path = "/Users/siddharthakosti/Downloads/catalyst_ragaai/RagaAI-Catalyst/ai_document_061023_2.pdf"
            
            # Create RAG pipeline
            qa_chain = create_rag_pipeline(pdf_path)
            # print(type(qa_chain))
            result = qa_chain.invoke("Provide specific title in 10 words about the doc")
            return result
        return main()
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/nano', methods=['POST'])
def chat_with_nano():
    """
    Endpoint to chat with GPT-4o-mini
    Expects JSON with 'message' field
    """
    try:
        tracer = Tracer(
            project_name="multi_endpoint_rag_tracer",
            dataset_name="gpt_nano_v1",
            tracer_type="langchain",
            )

        def create_rag_pipeline(pdf_path):
            # 1. Load the PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # 2. Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(documents)
            
            # 3. Create embeddings
            embeddings = OpenAIEmbeddings()
            
            # 4. Create vector store
            vectorstore = FAISS.from_documents(texts, embeddings)
            
            # 5. Create retrieval QA chain
            llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)
            qa_chain = RetrievalQA.from_chain_type(
                llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
            )
            
            return qa_chain

        def main():
            # Path to your PDF
            pdf_path = "/Users/siddharthakosti/Downloads/catalyst_ragaai/RagaAI-Catalyst/ai_document_061023_2.pdf"
            
            # Create RAG pipeline
            qa_chain = create_rag_pipeline(pdf_path)
            # print(type(qa_chain))
            result = qa_chain.invoke("Provide specific title in 20 words about the doc")
            return result
        return main()
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
