import os
# import sys
# sys.path.append('/Users/ragaai_user/work/ragaai-catalyst/')
import time
import json
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

import dotenv
dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import logging
import sys
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (INFO, DEBUG, ERROR, etc.)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Logs INFO and DEBUG to stdout
        logging.StreamHandler(sys.stderr),  # Logs WARNING and ERROR to stderr
    ],
)
logger = logging.getLogger("auto-impression")

from ragaai_catalyst import (
    Tracer, 
    RagaAICatalyst, 
    Evaluation
)
import pytest

access_key="1q2igAYCIlpSBufkdB6f" #os.getenv("RAGAAI_CATALYST_ACCESS_KEY")
secret_key="yG6TJOgES8D9jAi9OI0X6SgvZNtkcFvkOruukJay" #os.getenv("RAGAAI_CATALYST_SECRET_KEY")
base_url="https://llm-dev5.ragaai.ai/api" #os.getenv("RAGAAI_CATALYST_BASE_URL")


catalyst = RagaAICatalyst(
    access_key=access_key,
    secret_key=secret_key,
    base_url=base_url
)

def create_rag_pipeline(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
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

from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

def get_radiology_prompt(resp_dict: dict, format_instructions: str) -> ChatPromptTemplate:
    enhanced_instructions = f"""Output must strictly adhere to this format: {format_instructions}. Do not add any explanations or extra text"""
    impression_system = """You are an expert radiologist responsible for generating precise and focused impressions for radiology reports. Your impressions should adhere to current clinical guidelines and best practices from reputable radiological societies such as the ACR (American College of Radiology), RSNA (Radiological Society of North America), Fleischner Society, and other relevant organizations. Incorporate recommendations and guidelines from clinical papers and established frameworks to ensure optimal patient care.
    {enhanced_instructions}"""
    impression_prompt = """
    You are a knowledgeable radiologist responsible for crafting precise and focused impressions for radiology reports. Your impressions should be clear, concise, and adhere to the following guidelines.
        PRIORITY ORDER:
        Clinical Findings:
        Start with findings directly related to the reason for the exam or the primary clinical indications.

        Acute Findings Statement:
        Clearly state the presence or absence of acute findings in the examined regions.

        Post-Surgical Changes:
        Note any stable post-surgical changes relevant to the patient's history.

        Clinically Significant Findings:
        List clinically significant findings in order of importance, providing essential details.

        Unremarkable Exam Statement:
        If the entire exam findings are unremarkable and there are no clinical findings, simply state "Unremarkable exam." and do not include any other statements in the impression.

        CORE RULES:
        Exclude Normal and Negative Findings:
        Do not include normal findings or negative statements about structures without abnormalities unless directly addressing the clinical question.
        Avoid phrases like "No evidence of..." or "No significant abnormalities in...".

        Include Only Essential Findings:
        Focus on clinically significant findings that impact patient care or require follow-up.

        Include Recommendations When Necessary:
        Provide recommendations if they significantly impact patient management or are essential for follow-up.

        Avoid Redundant Phrases:
        Do not use phrases like "present," "noted," "identified," or "visualized."

        Exclude Unnecessary Details:
        Do not include technical details, unnecessary measurements, or incidental findings without clinical relevance.

        STYLE GUIDELINES:
        Numbered List:
        Always present the Impression as a numbered list, with each point numbered and on its own line for clarity.

        Conciseness:
        Keep the impression concise, using only as many lines as necessary for essential information.

        Order of Significance:
        List findings from most significant to least significant.

        Consistent Wording:
        Use clear and consistent terminology, avoiding unnecessary words or phrases.

        Punctuation:
        Use periods at the end of each statement.

        FORMAT SUMMARY:
        Clinical Findings:
        Address findings related to the clinical indications for the exam.

        Acute Findings Statement:
        State the presence or absence of acute findings relevant to the clinical concern.

        Post-Surgical Changes:
        Mention stable post-surgical changes if applicable.

        Clinically Significant Findings:
        Include significant findings with essential details, ordered by importance.

        Unremarkable Exam Statement:
        If the exam is completely unremarkable and there are no clinical findings, state "Unremarkable exam." and do not include any other statements in the impression.

        Your Task:
        Generate an IMPRESSION section based on the provided findings, strictly following these guidelines. Focus on conveying critical information to the referring clinician, ensuring the impression is concise and aligns with the known good impression in focus and content.

        Note:
        Only output the Impression text, presented as a numbered list.
        Do not include normal findings or unnecessary negative statements unless directly relevant to the clinical indication.
        Include significant recommendations if they impact patient management.
        If the exam is completely unremarkable and there are no clinical findings, state "Unremarkable exam." and do not include any other statements in the impression.
        Patient-Information:
        Age: {age}
        Gender: {gender}
        Ethnicity: {ethnicity}

        Radiography Exam:
        Type of Radiography Exam: {exam}
        Reason For Exam: {reason_for_exam}
        Technique: {technique}
        Indication Codes List: {indication_codes}

        Findings: {findings}

        """

    context = {
        "age": resp_dict.get("age"),
        "gender": resp_dict.get("gender"), "ethnicity": resp_dict.get("ethnicity"),
        "exam": resp_dict.get("exam"), "reason_for_exam": resp_dict.get("reason_for_exam"),
        "technique": resp_dict.get("technique"), "indication_codes": resp_dict.get("indication_codes"),
        "findings": resp_dict.get("findings"), "format_instructions": format_instructions
    }
    system_message = SystemMessagePromptTemplate.from_template(impression_system, partial_variables={"enhanced_instructions": enhanced_instructions})

    human_message = HumanMessagePromptTemplate.from_template(impression_prompt, partial_variables=context)

    prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    return prompt, context


from langchain_core.output_parsers import PydanticOutputParser
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain_google_vertexai import ChatVertexAI, VertexAI
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from pydantic import BaseModel, Field
import json
import logging

logger = logging.getLogger(__name__)

class ImpressionPromptOutput(BaseModel):
    impression: str = Field(description="Impression generated by LLM")

# Define model configurations
# Define model configurations
MODEL_CONFIGS = {
    # GCP Models
    "chat-vertex-gemini-1.5-flash-002": {
        "provider": "vertex",
        "model_class": ChatVertexAI,
        "kwargs": {
            "model": "gemini-1.5-flash-002",
            "project": "gen-lang-client-0655603261",
            "location": "us-central1",
        }
    },
    "chat-vertex-gemini-1.5-pro-002": {
        "provider": "google",
        "model_class": ChatVertexAI,
        "kwargs": {
            "model_name": "gemini-1.5-pro-002",
            "temperature": 0.7
        }
    },
    "vertex-gemini-1.5-flash-002": {
        "provider": "vertex",
        "model_class": VertexAI,
        "kwargs": {
            "model": "gemini-1.5-flash-002",
            "project": "gen-lang-client-0655603261",
            "location": "us-central1",
        }
    },
    "vertex-gemini-1.5-pro-002": {
        "provider": "google",
        "model_class": VertexAI,
        "kwargs": {
            "model_name": "gemini-1.5-pro-002",
            "temperature": 0.7
        }
    },

    "gemini-1.5-flash-002": {
        "provider": "vertex",
        "model_class": ChatGoogleGenerativeAI,
        "kwargs": {
            "model": "gemini-1.5-flash-002"
        }
    },
    "gemini-1.5-flash-002_streaming": {
        "provider": "vertex",
        "model_class": ChatGoogleGenerativeAI,
        "kwargs": {
            "model": "gemini-1.5-flash-002",
            "streaming": True
        }
    },    
    "gemini-1.5-pro-002": {
        "provider": "google",
        "model_class": ChatGoogleGenerativeAI,
        "kwargs": {
            "model": "gemini-1.5-pro-002",
            "temperature": 0.7
        }
    },
    "gemini-1.5-pro-002_streaming": {
        "provider": "google",
        "model_class": ChatGoogleGenerativeAI,
        "kwargs": {
            "model": "gemini-1.5-pro-002",
            "temperature": 0.7,
            "streaming": True
        }
    },

    # Anthropic Vertex
    "vertex-claude-3-5-sonnet-v2": {
        "provider": "vertex",
        "model_class": ChatAnthropicVertex,
        "kwargs": {
            "model_name": "claude-3-5-haiku@20241022"
        }
    },

    # Anthropic Models
    "claude-3-5-sonnet-latest": {
        "provider": "anthropic",
        "model_class": ChatAnthropic,
        "kwargs": {
            "model": "claude-3-5-sonnet-latest",
            "temperature": 0.7
        }
    },

    "claude-3-5-haiku-latest": {
        "provider": "anthropic",
        "model_class": ChatAnthropic,
        "kwargs": {
            "model": "claude-3-5-haiku-latest",
            "temperature": 0.7
        }
    },

    "claude-3-5-sonnet-latest_streaming": {
        "provider": "anthropic",
        "model_class": ChatAnthropic,
        "kwargs": {
            "model": "claude-3-5-sonnet-20240620",
            "temperature": 0.7,
            "streaming": True
        }
    },

    "claude-3-5-haiku-latest_streaming": {
        "provider": "anthropic",
        "model_class": ChatAnthropic,
        "kwargs": {
            "model": "claude-3-5-haiku-latest",
            "temperature": 0.7,
            "streaming": True
        }
    },

    # Bedrock Models
    "amazon.nova-pro": {
        "provider": "bedrock",
        "model_class": ChatBedrock,
        "kwargs": {
            "model_id": "amazon.nova-pro-v1:0",
            "temperature": 0.7,
            "region": "us-east-1"
        }
    },
    "amazon.nova-lite": {
        "provider": "bedrock",
        "model_class": ChatBedrock,
        "kwargs": {
            "model_id": "amazon.nova-lite-v1:0",
            "temperature": 0.7,
            "region": "us-east-1"
        }
    },
    "meta.llama3-70b": {
        "provider": "bedrock",
        "model_class": ChatBedrock,
        "kwargs": {
            "model_id": "meta.llama3-1-70b-instruct-v1:0",
            "temperature": 0.7,
            "region": "us-west-2"
        }
    },

    # Azure OpenAI Models
    "azure-gpt-4o": {
        "provider": "azure",
        "model_class": AzureChatOpenAI,
        "kwargs": {
            "openai_api_version": "2028-08-01-preview",
            "deployment_name": "gpt-4",
            "model_name": "gpt-4",
            "temperature": 0.7
        }
    },
    "azure-gpt-4o-mini": {
        "provider": "azure",
        "model_class": AzureChatOpenAI,
        "kwargs": {
            "openai_api_version": "2028-08-01-preview",
            "deployment_name": "gpt-4-mini",
            "model_name": "gpt-4-mini",
            "temperature": 0.7
        }
    },

    # OpenAI Models
    "gpt-4o": {
        "provider": "openai",
        "model_class": ChatOpenAI,
        "kwargs": {
            "model_name": "gpt-4o",
            "temperature": 0.7
        }
    },
    "gpt-4o_streaming": {
        "provider": "openai",
        "model_class": ChatOpenAI,
        "kwargs": {
            "model_name": "gpt-4o",
            "temperature": 0.7,
            "streaming": True
        }
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "model_class": ChatOpenAI,
        "kwargs": {
            "model_name": "gpt-4o-mini",
            "temperature": 0.7
        }
    },
    "gpt-4o-mini_streaming": {
        "provider": "openai",
        "model_class": ChatOpenAI,
        "kwargs": {
            "model_name": "gpt-4o-mini",
            "temperature": 0.7,
            "streaming": True
        }
    }
}

def generate(resp_dict, model_name):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}")

    config = MODEL_CONFIGS[model_name]
    model = config["model_class"](**config["kwargs"])
    parser = PydanticOutputParser(pydantic_object=ImpressionPromptOutput)
    prompt, context = get_radiology_prompt(resp_dict, parser.get_format_instructions())

    try:
        chain = prompt | model | parser
        output = chain.invoke({})
        print(f"Raw LLM Response: {output}")
    except Exception as e:
        logger.warning(f"Errors during chain execution: {e}")
        try:
            chain = prompt | model
            output = chain.invoke({})
            print(f"Raw LLM Response: here::::{output}")
            try:
                wrapped_resp = json.dumps({"impression": output})
            except:
                wrapped_resp = json.dumps({"impression": output.content})
            output = parser.parse(wrapped_resp)
        except Exception as fallback_error:
            logger.warning(f"Errors during chain fallback execution: {fallback_error}")
            output = None

    return output, context

def get_impression(resp_dict, model_name="gemini-1.5-flash-002"):
    new_impression, context = generate(resp_dict, model_name)
    return new_impression, context

resp_dict = {"tenant_id": "rRIS_NewCity", "study_key": 1026534339, "procedure_description": "CT Abdomen W", "reason_for_exam": "FAT LIVER Routine follow up in one year is recommended.", "ethnicity": "Not Hispanic or Latino", "gender": "Female", "indication_codes": [], "technique": "CT examination of the abdomen is obtained after the patient drank oral contrast and received intravenous administration of 100 cc of Optiray-350 low osmolar iodinated contrast material. Reformatted coronal views are provided with 3 mm collimation. One or more of the following dose reduction techniques were used: automated exposure control, adjustment of the mA and/or kV according to patient size, use of iterative reconstruction technique.", "findings": "The lung bases are clear.The liver, spleen, pancreas and left adrenal gland are unremarkable. There is a heterogeneous lesion within the right adrenal gland measuring 4.0 x 2.4 cm.The gallbladder is unremarkable.Both kidneys enhance symmetrically without renal calculi, hydronephrosis, or contour deforming solid mass.There is no mesenteric or retroperitoneal lymphadenopathy.There is mild aortic atherosclerotic disease without aortic aneurysm.The visualized bowel is unremarkable. The appendix is normal.No suspect osseous lesion is visualized.", "exam": "CT ABDOMEN WITH CONTRAST", "age": 73}

import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

GOOGLE_CLOUD_PROJECT="gen-lang-client-0655603261"
GOOGLE_CLOUD_REGION="us-central1"
GOOGLE_APPLICATION_CREDENTIALS="/Users/siddharthakosti/Downloads/catalyst_new_github_repo/gen-lang-client-0655603261-b28b82fe41b5_creds.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS



def run_pipeline():
    global tracer
    tracer = Tracer(
        project_name="tracing_check_sk",
        dataset_name="test_00",
        tracer_type='rag/langchain',
        metadata={
            "model": "gpt-3.5-turbo",
            "environment": "production"
        },
        pipeline={
            "llm_model": "gpt-3.5-turbo",
            "vector_store": "faiss",
            "embed_model": "text-embedding-ada-002",
        }
    )
    
    result, context = get_impression(resp_dict, "gpt-4o-mini")




class TestLangchainTracing:
    @classmethod
    def setup_class(cls):
        if os.path.exists('final_rag_traces.json'):
            os.remove('final_rag_traces.json')
        run_pipeline()
    
    @classmethod
    def teardown_class(cls):
        if os.path.exists('final_rag_traces.json'):
            os.remove('final_rag_traces.json')
    
    def test_final_result(self):
        assert os.path.exists('final_rag_traces.json'), "Final result file not created"
    
    def test_final_result_content(self):
        with open('final_rag_traces.json', 'r') as f:
            final_result = json.load(f)
        assert len(final_result) > 0, "Final result is empty"
    
    def test_traces_presence(self):
        with open('final_rag_traces.json', 'r') as f:
            final_result = json.load(f)
        assert 'traces' in final_result[0], "traces key not found in final result"
        traces = final_result[0]['traces']
        assert len(traces) > 0, "No traces found in final result"
    

    @pytest.mark.parametrize('part_name', [
        "retrieve_documents.langchain.workflow",
        "PromptTemplate.langchain.task", 
        "ChatOpenAI.langchain.task"
        ])
    def test_trace_parts(self, part_name):
        with open('final_rag_traces.json', 'r') as f:
            final_result = json.load(f)
        traces = final_result[0]['traces']
        parts = [trace['name'] for trace in traces]
        assert part_name in parts, f"{part_name} not found in final result"

    @pytest.mark.parametrize(('part_name', 'attr_len'), [
        ("retrieve_documents.langchain.workflow", 1),
        ("PromptTemplate.langchain.task", 2), 
        ("ChatOpenAI.langchain.task", 2)
        ])
    def test_traces(self, part_name, attr_len):
        with open('final_rag_traces.json', 'r') as f:
            final_result = json.load(f)
        traces = final_result[0]['traces']
        for trace in traces:
            if trace['name'] == part_name:
                assert len(trace['attributes']) == attr_len, f"{part_name} has incorrect number of attributes"

    def test_metadata_fields(self):
        """Test that all required metadata fields are present in the output."""
        with open('final_rag_traces.json', 'r') as f:
            final_result = json.load(f)
        
        required_metadata_fields = [
            'cost',
            'latency',
            'model_name',
            'prompt_tokens',
            'completion_tokens'
        ]
        
        metadata = final_result[0]['metadata']
        for field in required_metadata_fields:
            assert field in metadata, f"Required metadata field '{field}' not found"
            assert metadata[field] is not None, f"Metadata field '{field}' is None"

    def test_metadata_field_types(self):
        """Test that metadata fields have correct data types."""
        with open('final_rag_traces.json', 'r') as f:
            final_result = json.load(f)
        
        metadata = final_result[0]['metadata']
        
        assert isinstance(metadata['cost'], (int, float)), "Cost should be numeric"
        assert isinstance(metadata['latency'], (int, float)), "Latency should be numeric"
        assert isinstance(metadata['model_name'], str), "Model name should be string"
        assert isinstance(metadata['prompt_tokens'], int), "Prompt tokens should be integer"
        assert isinstance(metadata['completion_tokens'], int), "Completion tokens should be integer"
