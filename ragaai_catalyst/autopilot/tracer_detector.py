import ast
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Counter, Optional


class TracerTypeDetector:
    LIBRARY_TO_TRACER_TYPE = {
        "langchain": "langchain",
        "langgraph": "langgraph",
        "llama_index": "llamaindex",
        "crewai": "crewai",
        "smolagents": "smolagents",
        "haystack": "haystack",
        "agents": "openai_agents",
    }

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()
        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
        self.local_modules = self._get_local_modules()

    def _get_local_modules(self) -> set:
        modules = set()
        py_files = glob.glob(str(self.repo_path / "**" / "*.py"), recursive=True)
        for py_file in py_files:
            path = Path(py_file)
            modules.add(path.stem.lower())
        return modules

    def analyze_imports(self) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        detected_imports = {}
        framework_scores = Counter()

        py_files = glob.glob(str(self.repo_path / "**" / "*.py"), recursive=True)

        for file_path in py_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            module_name = name.name.split(".")[0].lower()
                            if module_name in self.local_modules:
                                continue
                            if module_name in self.LIBRARY_TO_TRACER_TYPE:
                                lib = module_name.lower()
                                if lib not in detected_imports:
                                    detected_imports[lib] = []
                                detected_imports[lib].append(file_path)

                                tracer_type = self.LIBRARY_TO_TRACER_TYPE[lib]
                                full_import = name.name

                                if "langchain" in full_import:
                                    framework_scores[tracer_type] += 1
                                else:
                                    framework_scores[tracer_type] += 3

                    elif isinstance(node, ast.ImportFrom):
                        module_name = (
                            node.module.split(".")[0] if node.module else ""
                        ).lower()
                        if module_name in self.local_modules:
                            continue
                        if module_name in self.LIBRARY_TO_TRACER_TYPE:
                            lib = module_name.lower()
                            if lib not in detected_imports:
                                detected_imports[lib] = []
                            detected_imports[lib].append(file_path)

                            tracer_type = self.LIBRARY_TO_TRACER_TYPE[lib]
                            full_import = (
                                f"{node.module}.{node.names[0].name}"
                                if node.module
                                else node.names[0].name
                            )

                            if "langchain" in full_import:
                                framework_scores[tracer_type] += 1
                            else:
                                framework_scores[tracer_type] += 3

            except Exception as e:
                print(f"Error parsing {file_path}: {e}")

        return framework_scores

    def analyze_repository(self) -> Tuple[str, Dict[str, int]]:
        framework_scores = self.analyze_imports()
        tracer_usage = Counter()

        for tracer_type, score in framework_scores.items():
            tracer_usage[tracer_type] += score

        if not tracer_usage:
            return "Agentic"

        suggested_tracer = max(tracer_usage.items(), key=lambda x: x[1])[0]
        return f"agentic/{suggested_tracer}"


def get_tracer_type(repo_path: str, usecase: Optional[str] = None) -> Dict[str, str]:
    try:
        detector = TracerTypeDetector(repo_path)
        suggested_tracer = detector.analyze_repository()
        project_type = "Agentic Application"

        if "llamaindex" or "langchain" in suggested_tracer:
            if usecase == "rag":
                project_type = "Q/A"
                suggested_tracer = suggested_tracer.split("/")[1]
        code = f"""
tracer = Tracer(
    project_name=os.environ['PROJECT_NAME'],
    dataset_name=os.environ['DATASET_NAME'],
    tracer_type="{suggested_tracer}",
    )
    """
        return {
            "suggested_tracer": suggested_tracer,
            "detected_frameworks": suggested_tracer.split("/")[0],
            "project_type": project_type,
            "code": code,
        }
    except Exception as e:
        print(f"Error analyzing repository: {e}")
