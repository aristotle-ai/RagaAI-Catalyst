import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from ragaai_catalyst import RagaAICatalyst
from ragaai_catalyst.tracers import Tracer,init_tracing

# from ragaai_catalyst import (
#     RagaAICatalyst,Tracer,
#     init_tracing
# )

import os
import re
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# catalyst = RagaAICatalyst(
#     access_key="T34UvqcxEB2Sk9ZHtXhF",
#     secret_key="R8yC7XyobB7BQL5mhguKmqECtquX7Q7f72A6o1Pr",
#     base_url="https://catalyst.raga.ai/api"
# )

catalyst = RagaAICatalyst(
    access_key="jv3B4OIxKY9pn2cYjXni",
    secret_key="1MPr3pEYDjlbQzPaLxbEAqskNHcdDHHrhmgfvY3b",
    base_url="http://4.240.58.193/api"
)

project_name = "qa_app"
dataset_name = "resgression_script_test1"


tracer = Tracer(
    project_name=project_name,
    dataset_name=dataset_name,
    metadata={"name": "Default", "age": 25},
    tracer_type="langchain",
    external_id="default_id"
)

def masking_function(value):
    symptoms = ['Union', 'Committee', 'Contract', 'Document', 'Secretary', 'Infrastructure', 'Monitoring', 'issues']
    for symptom in symptoms:
        value = re.sub(rf'\b{symptom}\b', '<REDACTED KEYWORD>', value, flags=re.IGNORECASE)

    return value

def another_masking_function(value):
    """
    Returns masked strings with dates and emails redacted
    """
    value = re.sub(r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{2}:\d{2}\b', '<REDACTED DATE>', value)
    value = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '<REDACTED DATE>', value)
    value = re.sub(r'\b\d{1,2}/\d{1,2}/\d{4}\b', '<REDACTED DATE>', value)
    value = re.sub(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', '< REDACTED EMAIL ADDRESS>', value)
    return value

def full_masking_function(value):
    value = '<REDACTED TEXT>'
    value = re.sub(r'.*', '<REDACTED TEXT>', value)
    return value

# tracer.register_masking_function(full_masking_function)
init_tracing(catalyst=catalyst, tracer=tracer)


def load_documents(data_dir="data"):
    loader = DirectoryLoader(
        path=data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    if not documents:
      raise ValueError("No documents found in %s" % data_dir)
    print(f"Loaded {len(documents)} documents")
    return documents

def split_documents(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    """Create vector store from document chunks"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Created vector store")
    return vector_store

def setup_rag_chain(vector_store):
    """Set up RAG chain with retrieval and generation"""
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    # llm = ChatGoogleGenerativeAI(
    #         model="gemini-1.5-flash-002",
    #         temperature=0

    #     )
    template = """
    You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Create retrieval chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

query_list = [
    "what is this paper about?",
    # "Does that paper have any relation with women empowerment?",
]

def main():
    """Main function to run the RAG application"""
    print("Loading documents...")
    documents = load_documents()

    print("Splitting documents...")
    chunks = split_documents(documents)

    print("Creating vector store...")
    vector_store = create_vector_store(chunks)

    print("Setting up RAG chain...")
    qa_chain = setup_rag_chain(vector_store)

    id_counter = 0
    name_options = ["alpha", "beta", "gamma", "delta", "epsilon"]
    current_name = name_options[0]


    for query in query_list:
      # UPDATE EXTERNAL ID
      id_counter+=1
      external_id = f"id_{id_counter}"
      tracer.set_external_id(external_id)

      # UPDATE METADATA
      current_name = name_options[id_counter % len(name_options)]
      tracer.add_metadata({"name": current_name, "age": id_counter})

      # UPDATE CONTEXT
      tracer.add_context("Sample Context")

      print("Generating answer...")
      # qa_chain = setup_rag_chain(vector_store)
      result = qa_chain({"query": query})

if __name__ == "__main__":
    main()
