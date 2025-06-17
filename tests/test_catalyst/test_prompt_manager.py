import os
import pytest
import copy
from ragaai_catalyst import PromptManager, RagaAICatalyst
import dotenv
import openai
import time
dotenv.load_dotenv()


@pytest.fixture
def base_url():
    return os.getenv("RAGAAI_CATALYST_BASE_URL")

@pytest.fixture
def access_keys():
    return {
        "access_key": os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
        "secret_key": os.getenv("RAGAAI_CATALYST_SECRET_KEY")}

@pytest.fixture(scope="session")
def create_test_project():
    """Create the test project if it doesn't exist already"""
    catalyst = RagaAICatalyst(
        access_key=os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
        secret_key=os.getenv("RAGAAI_CATALYST_SECRET_KEY"),
        base_url=os.getenv("RAGAAI_CATALYST_BASE_URL")
    )
    
    # Check if project already exists
    project_name = "prompt_metric_dataset"
    existing_projects = catalyst.list_projects()
    
    if project_name not in existing_projects:
        print(f"Creating test project: {project_name}")
        # Create the project - using "Chatbot" as a typical usecase
        result = catalyst.create_project(project_name=project_name, usecase="Chatbot")
        print(f"Project creation result: {result}")
        # Give the server some time to process the project creation
        time.sleep(2)
    else:
        print(f"Test project {project_name} already exists")
    
    return project_name

@pytest.fixture
def prompt_manager(base_url, access_keys, create_test_project):
    """Create evaluation instance with specific project and dataset"""
    os.environ["RAGAAI_CATALYST_BASE_URL"] = base_url
    catalyst = RagaAICatalyst(
        access_key=access_keys["access_key"],
        secret_key=access_keys["secret_key"]
    )
    return PromptManager(project_name=create_test_project)

def test_prompt_initialistaion(prompt_manager):
    prompt_list= prompt_manager.list_prompts()
    assert prompt_list ==['test','test2']

def test_list_prompt_version(prompt_manager):
    prompt_version_list = prompt_manager.list_prompt_versions(prompt_name="test2")
    assert len(prompt_version_list.keys()) == 3

def test_missing_prompt_name(prompt_manager):
    with pytest.raises(ValueError, match="Please enter a valid prompt name"):
        prompt = prompt_manager.get_prompt(prompt_name="", version="v1")

def test_get_variable(prompt_manager):
    prompt = prompt_manager.get_prompt(prompt_name="test2", version="v3")
    prompt_variable = prompt.get_variables()
    assert prompt_variable == ['system1', 'system2'] or prompt_variable == ['system2', 'system1']



def test_compile_prompt(prompt_manager):
    prompt = prompt_manager.get_prompt(prompt_name="test2", version="v3")
    compiled_prompt = prompt.compile(
    system1='What is chocolate?',
    system2 = "How it is made")
    def get_openai_response(prompt):
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt
        )
        return response.choices[0].message.content
    get_openai_response(compiled_prompt)

def test_compile_prompt_no_modelname(prompt_manager):
    with pytest.raises(openai.BadRequestError,match="you must provide a model parameter"):
        prompt = prompt_manager.get_prompt(prompt_name="test2", version="v3")
        compiled_prompt = prompt.compile(
        system1='What is chocolate?',
        system2 = "How it is made")
        def get_openai_response(prompt):
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="",
                messages=prompt
            )
            return response.choices[0].message.content
        get_openai_response(compiled_prompt)