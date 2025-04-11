import ast
import glob
from pathlib import Path
from typing import Dict, List, Counter, Optional


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
        self.import_locations = {}

    def _get_local_modules(self) -> set:
        modules = set()
        py_files = glob.glob(str(self.repo_path / "**" / "*.py"), recursive=True)
        for py_file in py_files:
            path = Path(py_file)
            modules.add(path.stem.lower())
        return modules

    def analyze_imports(self) -> Dict[str, int]:
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

                                self.import_locations.setdefault(
                                    tracer_type, set()
                                ).add(file_path)

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

                            self.import_locations.setdefault(tracer_type, set()).add(
                                file_path
                            )

            except Exception as e:
                print(f"Error parsing {file_path}: {e}")

        return framework_scores

    def find_best_insert_file(
        self, target_framework: Optional[str] = None
    ) -> Optional[str]:
        py_files = glob.glob(str(self.repo_path / "**" / "*.py"), recursive=True)
        scores = {}

        for file_path in py_files:
            score = 0
            file_name = Path(file_path).name.lower()

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    tree = ast.parse(content)

                    if "__main__" in content:
                        score += 3

                    if file_name in {"main.py", "app.py", "run.py"}:
                        score += 1

                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if node.name in {"run", "start", "main"}:
                                score += 2
                        if isinstance(node, ast.Expr) and isinstance(
                            node.value, ast.Call
                        ):
                            if hasattr(
                                node.value.func, "id"
                            ) and node.value.func.id in {"run", "start", "main"}:
                                score += 2

                    if target_framework:
                        if target_framework in content:
                            score += 1

                scores[file_path] = score

            except Exception:
                continue

        if not scores:
            return None

        return max(scores.items(), key=lambda x: x[1])[0]

    def update_file_with_tracer(self, file_path: str, tracer_code: str) -> bool:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")

            needs_os_import = "import os" not in content
            if needs_os_import:
                tracer_code = "import os\n\n" + tracer_code

            tree = ast.parse(content)
            import_nodes = []

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)) and hasattr(
                    node, "lineno"
                ):
                    if hasattr(node, "end_lineno"):
                        import_nodes.append((node.lineno, node.end_lineno))
                    else:
                        import_nodes.append((node.lineno, node.lineno))

            if not import_nodes:
                module_docstring = ast.get_docstring(tree)
                if module_docstring:
                    for node in tree.body:
                        if isinstance(node, ast.Expr) and isinstance(
                            node.value, ast.Constant
                        ):
                            if hasattr(node, "end_lineno"):
                                insert_line = node.end_lineno
                                break
                    else:
                        insert_line = 0
                else:
                    insert_line = 0
            else:
                import_nodes.sort()

                main_block_end = import_nodes[0][1]
                last_import_end = main_block_end

                for i in range(1, len(import_nodes)):
                    current_start, current_end = import_nodes[i]
                    if current_start <= last_import_end + 3:
                        main_block_end = current_end
                        last_import_end = current_end
                    else:
                        break

                insert_line = main_block_end

                paren_depth = 0
                for i in range(
                    max(0, insert_line - 10), min(len(lines), insert_line + 1)
                ):
                    paren_depth += lines[i].count("(") - lines[i].count(")")
                    if lines[i].strip().endswith("\\"):
                        insert_line = i + 1

                while paren_depth > 0 and insert_line < len(lines):
                    paren_depth += lines[insert_line].count("(") - lines[
                        insert_line
                    ].count(")")
                    insert_line += 1

                if insert_line < len(lines) and lines[insert_line].strip() == ")":
                    insert_line += 1

            new_lines = (
                lines[:insert_line] + ["", tracer_code, ""] + lines[insert_line:]
            )

            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines))

            return True

        except Exception as e:
            print(f"Error updating file {file_path}: {e}")
            return False


def get_tracer_type(
    repo_path: str, usecase: Optional[str] = None, auto_update: bool = False
) -> List[Dict[str, str]]:
    try:
        detector = TracerTypeDetector(repo_path)
        framework_scores = detector.analyze_imports()
        results = []

        if not framework_scores:
            code = """
#from ragaai_catalyst import RagaAICatalyst, init_tracing
#from ragaai_catalyst.tracers import Tracer
#catalyst = RagaAICatalyst(
#    access_key=os.getenv('CATALYST_ACCESS_KEY'), 
#    secret_key=os.getenv('CATALYST_SECRET_KEY'), 
#    base_url=os.getenv('CATALYST_BASE_URL')
#)
#tracer = Tracer(
#    project_name=os.environ['PROJECT_NAME'],
#    dataset_name=os.environ['DATASET_NAME'],
#    tracer_type="agentic",
#)
#init_tracing(catalyst=catalyst, tracer=tracer)
            """.strip()
            suggested_file = detector.find_best_insert_file()

            result = {
                "suggested_tracer": "agentic",
                "detected_frameworks": "agentic",
                "project_type": "Agentic Application",
                "code": code,
                "suggested_file": suggested_file or "main.py",
            }

            if auto_update and suggested_file:
                update_successful = detector.update_file_with_tracer(
                    suggested_file, code
                )
                result["update_status"] = (
                    "Updated successfully" if update_successful else "Update failed"
                )

            results.append(result)
            return results

        sorted_tracers = sorted(
            framework_scores.items(), key=lambda x: x[1], reverse=True
        )

        update_done = False

        for tracer_type, _ in sorted_tracers:
            project_type = "Agentic Application"
            if tracer_type in {"llamaindex", "langchain"} and usecase == "rag":
                project_type = "Q/A"

            suggested_tracer = f"agentic/{tracer_type}"
            code = f"""
#from ragaai_catalyst import RagaAICatalyst, init_tracing
#from ragaai_catalyst.tracers import Tracer
#catalyst = RagaAICatalyst(
#    access_key=os.getenv('CATALYST_ACCESS_KEY'), 
#    secret_key=os.getenv('CATALYST_SECRET_KEY'), 
#    base_url=os.getenv('CATALYST_BASE_URL')
#)
#tracer = Tracer(
#    project_name=os.environ['PROJECT_NAME'],
#    dataset_name=os.environ['DATASET_NAME'],
#    tracer_type="{suggested_tracer}",
#)
#init_tracing(catalyst=catalyst, tracer=tracer)
            """.strip()

            suggested_file = detector.find_best_insert_file(tracer_type)

            result = {
                "suggested_tracer": suggested_tracer,
                "detected_frameworks": tracer_type,
                "project_type": project_type,
                "code": code,
                "suggested_file": suggested_file or "main.py",
            }

            if auto_update and suggested_file and not update_done:
                update_successful = detector.update_file_with_tracer(
                    suggested_file, code
                )
                result["update_status"] = (
                    "Updated successfully" if update_successful else "Update failed"
                )
                update_done = True

            results.append(result)

        return results

    except Exception as e:
        print(f"Error analyzing repository: {e}")
        return []
