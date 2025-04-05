from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def initialize_tracing():
    catalyst = RagaAICatalyst(
        access_key=os.getenv("CATALYST_ACCESS_KEY"),
        secret_key=os.getenv("CATALYST_SECRET_KEY"),
        base_url=os.getenv("CATALYST_BASE_URL"),
    )

    tracer = Tracer(
        project_name=os.getenv("PROJECT_NAME"),
        dataset_name=os.getenv("DATASET_NAME"),
        tracer_type="Agentic",
    )

    init_tracing(catalyst=catalyst, tracer=tracer)
    return tracer
