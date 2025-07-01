import os
from ragaai_catalyst.session_manager import session_manager
import json
import re
from .ragaai_catalyst import RagaAICatalyst
import copy
import logging
import time
from urllib3.exceptions import PoolError, MaxRetryError, NewConnectionError
from requests.exceptions import ConnectionError, Timeout, RequestException
from http.client import RemoteDisconnected

logger = logging.getLogger(__name__)

class PromptManager:
    NUM_PROJECTS = 100
    TIMEOUT = 10

    def __init__(self, project_name):
        """
        Initialize the PromptManager with a project name.

        Args:
            project_name (str): The name of the project.

        Raises:
            ValueError: If the project is not found.
        """
        self.project_name = project_name
        self.base_url = f"{RagaAICatalyst.BASE_URL}/playground/prompt"
        self.timeout = 10
        self.size = 99999 #Number of projects to fetch
        self.project_id = None
        self.headers = {}

        try:
            start_time = time.time()
            response = session_manager.make_request_with_retry(
                "GET",
                f"{RagaAICatalyst.BASE_URL}/v2/llm/projects?size={self.size}",
                headers={
                    "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                },
                timeout=self.timeout,
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"API Call: [GET] /v2/llm/projects | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")

            if response.status_code in [200, 201]:
                # logger.debug("Projects list retrieved successfully")
                project_list = [
                    project["name"] for project in response.json()["data"]["content"]
                ]

                # Check if project exists before trying to get its ID
                if project_name not in project_list:
                    logger.error(f"Project '{project_name}' not found. Available projects: {project_list}")
                    return

                matching_projects = [
                    project["id"] for project in response.json()["data"]["content"]
                    if project["name"] == project_name
                ]
                if matching_projects:
                    self.project_id = matching_projects[0]
                else:
                    logger.error(f"Project '{project_name}' not found in project list")
                    return
            elif response.status_code == 401:
                logger.warning("Received 401 error during fetching project list. Attempting to refresh token.")
                token = RagaAICatalyst.get_token(force_refresh=True)
                headers = {
                    "Authorization": f"Bearer {token}",
                }
                start_time = time.time()
                response = session_manager.make_request_with_retry(
                    "GET", f"{RagaAICatalyst.BASE_URL}/v2/llm/projects?size={self.size}", 
                    headers=headers, timeout=self.timeout
                )
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [GET] /v2/llm/projects (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")

                if response.status_code in [200, 201]:
                    logger.info("Project list fetched successfully after 401 token refresh")
                    project_list = [
                        project["name"] for project in response.json()["data"]["content"]
                    ]

                    # Check if project exists before trying to get its ID
                    if project_name not in project_list:
                        logger.error(f"Project '{project_name}' not found. Available projects: {project_list}")
                        return

                    # Safe assignment now that we know project exists
                    matching_projects = [
                        project["id"] for project in response.json()["data"]["content"]
                        if project["name"] == project_name
                    ]
                    if matching_projects:
                        self.project_id = matching_projects[0]
                    else:
                        logger.error(f"Project '{project_name}' not found in project list")
                        return
                else:
                    logger.error("Failed to fetch project list after 401 token refresh")
                    return
            else:
                logger.error(f"HTTP {response.status_code} error when fetching project list")
                return

        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, "fetching project list")
            logger.error(f"Failed to fetch project list, PromptManager will have limited functionality")
            return
        except RequestException as e:
            logger.error(f"Error while fetching project list: {e}")
            logger.error(f"PromptManager will have limited functionality")
            return
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing project list: {str(e)}")
            return

        except Exception as e:
            logger.error(f"Unexpected error during project initialization: {str(e)}")
            return

        # Create headers for subsequent API calls
        self.headers = {
                "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                "X-Project-Id": str(self.project_id)
            }


    def list_prompts(self):
        """
        List all available prompts.

        Returns:
            list: A list of prompt names, or empty list if error occurs.
        """
        if not self.project_id:
            logger.error("PromptManager not properly initialized, cannot list prompts")
            return []
            
        prompt = Prompt()
        try:
            prompt_list = prompt.list_prompts(self.base_url, self.headers, self.timeout)
            return prompt_list
        except Exception as e:
            logger.error(f"Error listing prompts: {str(e)}")
            return []
    
    def get_prompt(self, prompt_name, version=None):
        """
        Get a specific prompt.

        Args:
            prompt_name (str): The name of the prompt.
            version (str, optional): The version of the prompt. Defaults to None.

        Returns:
            PromptObject: An object representing the prompt, or None if error occurs.
        """
        if not self.project_id:
            logger.error("PromptManager not properly initialized, cannot get prompt")
            return None
            
        try:
            prompt_list = self.list_prompts()
        except Exception as e:
            logger.error(f"Error fetching prompt list: {str(e)}")
            return None

        if prompt_name not in prompt_list:
            logger.error("Prompt not found. Please enter a valid prompt name")
            return None

        try:
            prompt_versions = self.list_prompt_versions(prompt_name)
        except Exception as e:
            logger.error(f"Error fetching prompt versions: {str(e)}")
            return None

        if version and version not in prompt_versions.keys():
            logger.error("Version not found. Please enter a valid version name")
            return None

        prompt = Prompt()
        try:
            prompt_object = prompt.get_prompt(self.base_url, self.headers, self.timeout, prompt_name, version)
            return prompt_object
        except Exception as e:
            logger.error(f"Error fetching prompt: {str(e)}")
            return None

    def list_prompt_versions(self, prompt_name):
        """
        List all versions of a specific prompt.

        Args:
            prompt_name (str): The name of the prompt.

        Returns:
            dict: A dictionary mapping version names to prompt texts, or empty dict if error occurs.
        """
        if not self.project_id:
            logger.error("PromptManager not properly initialized, cannot list prompt versions")
            return {}
            
        try:
            prompt_list = self.list_prompts()
        except Exception as e:
            logger.error(f"Error fetching prompt list: {str(e)}")
            return {}

        if prompt_name not in prompt_list:
            logger.error("Prompt not found. Please enter a valid prompt name")
            return {}
        
        prompt = Prompt()
        try:
            prompt_versions = prompt.list_prompt_versions(self.base_url, self.headers, self.timeout, prompt_name)
            return prompt_versions
        except Exception as e:
            logger.error(f"Error fetching prompt versions: {str(e)}")
            return {}


class Prompt:
    def __init__(self):
        """
        Initialize the Prompt class.
        """
        pass

    def list_prompts(self, url, headers, timeout):
        """
        List all available prompts.

        Args:
            url (str): The base URL for the API.
            headers (dict): The headers to be used in the request.
            timeout (int): The timeout for the request.

        Returns:
            list: A list of prompt names, or empty list if error occurs.
        """
        try:
            start_time = time.time()
            response = session_manager.make_request_with_retry("GET", url, headers=headers, timeout=timeout)
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"API Call: [GET] {url} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")

            if response.status_code in [200, 201]:
                prompt_list = [prompt["name"] for prompt in response.json()["data"]]
                return prompt_list
            elif response.status_code == 401:
                logger.warning("Received 401 error during listing prompts. Attempting to refresh token.")
                token = RagaAICatalyst.get_token(force_refresh=True)
                new_headers = headers.copy()
                new_headers["Authorization"] = f"Bearer {token}"

                start_time = time.time()
                response = session_manager.make_request_with_retry("GET", url, headers=new_headers, timeout=timeout)
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [GET] {url} (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")

                if response.status_code in [200, 201]:
                    logger.info("Prompts listed successfully after 401 token refresh")
                    prompt_list = [prompt["name"] for prompt in response.json()["data"]]
                    return prompt_list
                else:
                    logger.error("Failed to list prompts after 401 token refresh")
                    return []
            else:
                logger.error(f"HTTP {response.status_code} error when listing prompts")
                return []

        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, "listing prompts")
            return []
        except RequestException as e:
            logger.error(f"Error while listing prompts: {e}")
            return []
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing prompt list: {str(e)}")
            return []

    def _get_response_by_version(self, base_url, headers, timeout, prompt_name, version):
        """
        Get a specific version of a prompt.

        Args:
            base_url (str): The base URL for the API.
            headers (dict): The headers to be used in the request.
            timeout (int): The timeout for the request.
            prompt_name (str): The name of the prompt.
            version (str): The version of the prompt.

        Returns:
            response: The response object containing the prompt version data, or None if error occurs.
        """
        try:
            url = f"{base_url}/version/{prompt_name}?version={version}"
            start_time = time.time()
            response = session_manager.make_request_with_retry("GET", url, headers=headers, timeout=timeout)
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"API Call: [GET] {url} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")

            if response.status_code in [200, 201]:
                return response
            elif response.status_code == 401:
                logger.warning(f"Received 401 error during fetching prompt version {version} for {prompt_name}. Attempting to refresh token.")
                token = RagaAICatalyst.get_token(force_refresh=True)
                new_headers = headers.copy()
                new_headers["Authorization"] = f"Bearer {token}"

                start_time = time.time()
                response = session_manager.make_request_with_retry("GET", url, headers=new_headers, timeout=timeout)
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [GET] {url} (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")

                if response.status_code in [200, 201]:
                    logger.info(f"Prompt version {version} for {prompt_name} fetched successfully after 401 token refresh")
                    return response
                else:
                    logger.error(f"Failed to fetch prompt version {version} for {prompt_name} after 401 token refresh")
                    return None
            else:
                logger.error(f"HTTP {response.status_code} error when fetching prompt version {version} for {prompt_name}")
                return None

        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, f"fetching prompt version {version} for {prompt_name}")
            return None
        except RequestException as e:
            logger.error(f"Error while fetching prompt version {version} for {prompt_name}: {e}")
            return None
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            logger.error(f"Error parsing prompt version: {str(e)}")
            return None

    def _get_response(self, base_url, headers, timeout, prompt_name):
        """
        Get the latest version of a prompt.

        Args:
            base_url (str): The base URL for the API.
            headers (dict): The headers to be used in the request.
            timeout (int): The timeout for the request.
            prompt_name (str): The name of the prompt.

        Returns:
            response: The response object containing the latest prompt version data, or None if error occurs.
        """
        try:
            url = f"{base_url}/version/{prompt_name}"
            start_time = time.time()
            response = session_manager.make_request_with_retry("GET", url, headers=headers, timeout=timeout)
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"API Call: [GET] {url} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")

            if response.status_code in [200, 201]:
                return response
            elif response.status_code == 401:
                logger.warning(f"Received 401 error during fetching latest prompt version for {prompt_name}. Attempting to refresh token.")
                token = RagaAICatalyst.get_token(force_refresh=True)
                new_headers = headers.copy()
                new_headers["Authorization"] = f"Bearer {token}"

                start_time = time.time()
                response = session_manager.make_request_with_retry("GET", url, headers=new_headers, timeout=timeout)
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [GET] {url} (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")

                if response.status_code in [200, 201]:
                    logger.info(f"Latest prompt version for {prompt_name} fetched successfully after 401 token refresh")
                    return response
                else:
                    logger.error(f"Failed to fetch latest prompt version for {prompt_name} after 401 token refresh")
                    return None
            else:
                logger.error(f"HTTP {response.status_code} error when fetching latest prompt version for {prompt_name}")
                return None

        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, f"fetching latest prompt version for {prompt_name}")
            return None
        except RequestException as e:
            logger.error(f"Error while fetching latest prompt version for {prompt_name}: {e}")
            return None
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            logger.error(f"Error parsing prompt version: {str(e)}")
            return None

    def _get_prompt_by_version(self, base_url, headers, timeout, prompt_name, version):
        """
        Get a specific version of a prompt.

        Args:
            base_url (str): The base URL for the API.
            headers (dict): The headers to be used in the request.
            timeout (int): The timeout for the request.
            prompt_name (str): The name of the prompt.
            version (str): The version of the prompt.

        Returns:
            str: The text of the prompt, or empty string if error occurs.
        """
        response = self._get_response_by_version(base_url, headers, timeout, prompt_name, version)
        if response is None:
            return ""
        try:
            prompt_text = response.json()["data"]["docs"][0]["textFields"]
            return prompt_text
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            logger.error(f"Error parsing prompt text: {str(e)}")
            return ""

    def get_prompt(self, base_url, headers, timeout, prompt_name, version=None):
        """
        Get a prompt, optionally specifying a version.

        Args:
            base_url (str): The base URL for the API.
            headers (dict): The headers to be used in the request.
            timeout (int): The timeout for the request.
            prompt_name (str): The name of the prompt.
            version (str, optional): The version of the prompt. Defaults to None.

        Returns:
            PromptObject: An object representing the prompt, or None if error occurs.
        """
        if version:
            response = self._get_response_by_version(base_url, headers, timeout, prompt_name, version)
        else:
            response = self._get_response(base_url, headers, timeout, prompt_name)
            
        if response is None:
            return None

        try:
            prompt_text = response.json()["data"]["docs"][0]["textFields"]
            prompt_parameters = response.json()["data"]["docs"][0]["modelSpecs"]["parameters"]
            model = response.json()["data"]["docs"][0]["modelSpecs"]["model"]
            return PromptObject(prompt_text, prompt_parameters, model)
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            logger.error(f"Error parsing prompt data: {str(e)}")
            return None


    def list_prompt_versions(self, base_url, headers, timeout, prompt_name):
        """
        List all versions of a specific prompt.

        Args:
            base_url (str): The base URL for the API.
            headers (dict): The headers to be used in the request.
            timeout (int): The timeout for the request.
            prompt_name (str): The name of the prompt.

        Returns:
            dict: A dictionary mapping version names to prompt texts, or empty dict if error occurs.
        """
        try:
            url = f"{base_url}/{prompt_name}/version"
            start_time = time.time()
            response = session_manager.make_request_with_retry("GET", url, headers=headers, timeout=timeout)
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"API Call: [GET] {url} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")

            if response.status_code in [200, 201]:
                version_names = [version["name"] for version in response.json()["data"]]
                prompt_versions = {}
                for version in version_names:
                    prompt_versions[version] = self._get_prompt_by_version(base_url, headers, timeout, prompt_name, version)
                return prompt_versions
            elif response.status_code == 401:
                logger.warning(f"Received 401 error during listing prompt versions for {prompt_name}. Attempting to refresh token.")
                token = RagaAICatalyst.get_token(force_refresh=True)
                new_headers = headers.copy()
                new_headers["Authorization"] = f"Bearer {token}"

                start_time = time.time()
                response = session_manager.make_request_with_retry("GET", url, headers=new_headers, timeout=timeout)
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [GET] {url} (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")

                if response.status_code in [200, 201]:
                    logger.info(f"Prompt versions for {prompt_name} listed successfully after 401 token refresh")
                    version_names = [version["name"] for version in response.json()["data"]]
                    prompt_versions = {}
                    for version in version_names:
                        prompt_versions[version] = self._get_prompt_by_version(base_url, new_headers, timeout, prompt_name, version)
                    return prompt_versions
                else:
                    logger.error(f"Failed to list prompt versions for {prompt_name} after 401 token refresh")
                    return {}
            else:
                logger.error(f"HTTP {response.status_code} error when listing prompt versions for {prompt_name}")
                return {}

        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, f"listing prompt versions for {prompt_name}")
            return {}
        except RequestException as e:
            logger.error(f"Error while listing prompt versions for {prompt_name}: {e}")
            return {}
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing prompt versions: {str(e)}")
            return {}


class PromptObject:
    def __init__(self, text, parameters, model):
        """
        Initialize a PromptObject with the given text.

        Args:
            text (str): The text of the prompt.
            parameters (dict): The parameters of the prompt.
            model (str): The model of the prompt.
        """
        self.text = text
        self.parameters = parameters
        self.model = model

    def _extract_variable_from_content(self, content):
        """
        Extract variables from the content.

        Args:
            content (str): The content containing variables.

        Returns:
            list: A list of variable names found in the content.
        """
        pattern = r'\{\{(.*?)\}\}'
        matches = re.findall(pattern, content)
        variables = [match.strip() for match in matches if '"' not in match]
        return variables

    def _add_variable_value_to_content(self, content, user_variables):
        """
        Add variable values to the content.

        Args:
            content (str): The content containing variables.
            user_variables (dict): A dictionary of user-provided variable values.

        Returns:
            str: The content with variables replaced by their values.
        """
        variables = self._extract_variable_from_content(content)
        for key, value in user_variables.items():
            if not isinstance(value, str):
                raise ValueError(f"Value for variable '{key}' must be a string, not {type(value).__name__}")
            if key in variables:
                content = content.replace(f"{{{{{key}}}}}", value)
        return content

    def compile(self, **kwargs):
        """
        Compile the prompt by replacing variables with provided values.

        Args:
            **kwargs: Keyword arguments where keys are variable names and values are their replacements.

        Returns:
            str: The compiled prompt with variables replaced.

        Raises:
            ValueError: If there are missing or extra variables, or if a value is not a string.
        """
        required_variables = self.get_variables()
        provided_variables = set(kwargs.keys())

        missing_variables = [item for item in required_variables if item not in provided_variables]
        extra_variables = [item for item in provided_variables if item not in required_variables]

        if missing_variables:
            raise ValueError(f"Missing variable(s): {', '.join(missing_variables)}")
        if extra_variables:
            raise ValueError(f"Extra variable(s) provided: {', '.join(extra_variables)}")

        updated_text = copy.deepcopy(self.text)

        for item in updated_text:
            item["content"] = self._add_variable_value_to_content(item["content"], kwargs)

        return updated_text
    
    def get_variables(self):
        """
        Get all variables in the prompt text.

        Returns:
            list: A list of variable names found in the prompt text.
        """
        try:
            variables = set()
            for item in self.text:
                content = item["content"]
                for var in self._extract_variable_from_content(content):
                    variables.add(var)
            if variables:
                return list(variables)
            else:
                return []
        except (KeyError, TypeError, AttributeError) as e:
            logger.error(f"Error extracting variables: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in get_variables: {str(e)}")
            return []
    
    def _convert_value(self, value, type_):
        """
        Convert value based on type.

        Args:
            value: The value to be converted.
            type_ (str): The type to convert the value to.

        Returns:
            The converted value.
        """
        if type_ == "float":
            return float(value)
        elif type_ == "int":
            return int(value)
        return value  # Default case, return as is

    def get_model_parameters(self):
        """
        Get all parameters in the prompt text.

        Returns:
            dict: A dictionary of parameters found in the prompt text.
        """
        parameters = {}
        for param in self.parameters:
            if "value" in param:
                parameters[param["name"]] = self._convert_value(param["value"], param["type"])
            else:
                parameters[param["name"]] = ""
        parameters["model"] = self.model
        return parameters    
    
    def get_prompt_content(self):
        return self.text
