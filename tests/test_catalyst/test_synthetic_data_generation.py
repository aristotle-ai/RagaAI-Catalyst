# import sys
# sys.path.append('/Users/ritikagoel/workspace/synthetic-catalyst-internal-api2/ragaai-catalyst')

import pytest
from ragaai_catalyst import SyntheticDataGeneration
import os

import dotenv
dotenv.load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

doc_path = os.path.join(os.path.dirname(__file__), os.path.join("test_data", "util_synthetic_data_doc.csv"))
valid_csv_path = os.path.join(os.path.dirname(__file__), os.path.join("test_data", "util_synthetic_data_valid.csv"))
invalid_csv_path = os.path.join(os.path.dirname(__file__), os.path.join("test_data", "util_synthetic_data_invalid.csv"))

@pytest.fixture
def synthetic_gen():
    return SyntheticDataGeneration()

@pytest.fixture
def sample_text(synthetic_gen):
    text_file = doc_path # Update this path as needed
    return synthetic_gen.process_document(input_data=text_file)

def test_special_chars_csv_processing(synthetic_gen):
    """Test processing CSV with special characters"""
    with pytest.raises(Exception):
        synthetic_gen.process_document(input_data=valid_csv_path)
    

def test_invalid_llm_proxy(synthetic_gen, sample_text):
    """Test behavior with invalid internal_llm_proxy URL"""
    with pytest.raises(Exception, match="No connection adapters were found for"):
        synthetic_gen.generate_qna(
            text=sample_text,
            question_type='mcq',
            model_config={"provider": "openai", "model": "gpt-4o-mini"},
            n=1,
            internal_llm_proxy="tp://invalid.url",
            user_id="1"
        )

def test_missing_model_config(synthetic_gen, sample_text):
    """Test behavior when model_config is not provided"""
    with pytest.raises(ValueError, match="Model configuration must be provided with a valid provider and model"):
        synthetic_gen.generate_qna(
            text=sample_text,
            question_type='mcq',
            n=1,
            internal_llm_proxy="http://20.244.126.4:4000/chat/completions",
            user_id="1"
        )
