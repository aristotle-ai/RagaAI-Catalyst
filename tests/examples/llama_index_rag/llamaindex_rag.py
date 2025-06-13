import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from ragaai_catalyst import RagaAICatalyst
from ragaai_catalyst.tracers import Tracer,init_tracing
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.readers.file import PDFReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import logging
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

catalyst = RagaAICatalyst(
    access_key="hShxuaTU4DmkTLPziBcT",
    secret_key="kgRue1n59Oq2jkATPggNeHHPorVpHXz2AG6NKY1Y",
    base_url="http://4.240.58.193/api"
)

project_name = "tracing_check_sk"
tracer_dataset_name = "llamaindex_rag_v3"

tracer = Tracer(
    project_name=project_name,
    dataset_name=tracer_dataset_name,
    tracer_type="llamaindex",
)


init_tracing(catalyst=catalyst, tracer=tracer)

retriever = None
loaded_doc = None
index = None

def load_document(source_doc_path):
    """
    Load and index the document using LlamaIndex
    """
    try:
        # Initialize LLM and embedding model
        Settings.llm = OpenAI(model="gpt-4o-mini")
        Settings.embed_model = OpenAIEmbedding()


        # Load PDF document
        reader = PDFReader()
        docs = reader.load_data(source_doc_path)

        # Create documents with metadata
        documents = [
            Document(text=doc.text, metadata={"source": source_doc_path})
            for doc in docs
        ]

        # Create vector store index
        global index
        index = VectorStoreIndex.from_documents(documents)

        # Create retriever (to maintain similar interface)
        retriever = index.as_retriever(similarity_top_k=5)

        logger.info("Document loaded and processed.")
        return retriever

    except Exception as e:
        logger.error(f"An error occurred while loading the document: {e}")
        return None

def generate_response(retriever, query):
    """
    Generate response for the given query using LlamaIndex
    """
    try:
        if index is None:
            logger.error("Index not initialized. Please load document first.")
            return None

        # Create query engine
        query_engine = index.as_query_engine(
            response_mode="compact"
        )

        # Generate response
        response = query_engine.query(query)

        logger.info("Response generated successfully")
        return str(response)

    except Exception as e:
        logger.error(f"An error occurred while generating the response: {e}")
        return None

def llamaindex_rag(source_doc_path, loaded_doc, query):
    """
    Process document and generate response using LlamaIndex
    """
    try:
        # Check if we need to load a new document
        if loaded_doc != source_doc_path:
            retriever = load_document(source_doc_path)
            if retriever is None:
                return "Failed to load document."
            loaded_doc = source_doc_path
        else:
            logger.info("Using cached document retriever.")

        # Generate response
        response = generate_response(retriever, query)
        if response is None:
            return "Failed to generate response."

        return response

    except Exception as e:
        logger.error(f"An overall error occurred: {e}")
        return "An error occurred during the document processing."



source_doc_path = "/Users/ritikagoel/workspace/regression_pytest_github/RagaAI-Catalyst/tests/examples/llama_index_rag/data/AI_Introduction.pdf"

# Process a query
if __name__ == "__main__":
    query = "what is this paper about?"
    response = llamaindex_rag(source_doc_path, None, query)
    print(f"Response: {response}")
