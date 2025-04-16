# === Standard Library ===
import os
import sys
import argparse

# === Third-Party Libraries ===
from dotenv import load_dotenv
import pytest

# === LangChain & Related ===
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# === OpenAI + RagaAI ===
from openai import OpenAI
from ragaai_catalyst import RagaAICatalyst, Tracer

# === Setup ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Raga Catalyst Initialization ===
catalyst = RagaAICatalyst(
    access_key=os.getenv("RAGAAI_CATALYST_ACCESS_KEY_DEV5"),
    secret_key=os.getenv("RAGAAI_CATALYST_SECRET_KEY_DEV5"),
    base_url="https://llm-dev5.ragaai.ai/api"
)

tracer = Tracer(
    project_name="externalID",
    dataset_name="testing_eid",
    external_id="my_external_id",
    metadata={"key1": "value1", "key2": "value2"},
    tracer_type="langchain",
    pipeline={
        "llm_model": "gpt-4o-mini",
        "vector_store": "faiss",
        "embed_model": "text-embedding-ada-002",
    }
)

# === RAG Pipeline ===
def create_rag_pipeline(pdf_path):
    """Creates a LangChain RAG pipeline from a given PDF."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
    )
    return qa_chain

def call_rag_pipeline(question):
    DIR = os.path.dirname(__file__)
    pdf_path = os.path.join(DIR, "..", "father_forgets.pdf")
    qa_chain = create_rag_pipeline(pdf_path)
    return qa_chain.invoke(question)

def main_normal_external_id_check_run(query):
    response = call_rag_pipeline(query)
    print(response)

def main_dynamic_external_id_check_run(query, external_id):
    tracer.set_external_id(external_id)
    response = call_rag_pipeline(query)
    print(response)

# === CLI Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the external_id functionality.")
    parser.add_argument("--query", type=str, default="summarise the doc", help="The question to ask.")
    parser.add_argument("--external_id", type=str, default="my_external_id", help="External ID for trace.")
    args = parser.parse_args()

    if args.external_id == "my_external_id":
        # Run with the default external ID
        main_normal_external_id_check_run(args.query)
    else:
        # Run with a dynamic external ID
        main_dynamic_external_id_check_run(args.query, args.external_id)


# === Test Suite ===
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from examples.test_utils.get_trace_data import (
    run_command,
    extract_information,
    load_trace_data
)

@pytest.mark.parametrize("query", [
    ("What is the main theme of the document?"),
])
def test_normal_external_id_check_run(query: str):
    """Integration test for external ID tracing."""
    command = f'python test_external_id.py --query "{query}"'
    cwd = os.path.dirname(os.path.abspath(__file__))
    output = run_command(command, cwd=cwd)

    locations = extract_information(output)
    data = load_trace_data(locations)

    assert data[0]['external_id'] == 'my_external_id', f"Expected data[0]['external_id'] = 'my_external_id', got {data}"



@pytest.mark.parametrize("query,external_id", [
    ("Summarise the document in 5 lines", "eid1"),
])
def test_dynamic_external_id_check_run(query: str, external_id: str):
    """Integration test for external ID tracing."""
    command = f'python test_external_id.py --query "{query}" --external_id {external_id}'
    cwd = os.path.dirname(os.path.abspath(__file__))
    output = run_command(command, cwd=cwd)

    locations = extract_information(output)
    data = load_trace_data(locations)

    assert data[0]['external_id'] == 'eid1', f"Expected data[0]['external_id'] = 'eid1', got {data}"
