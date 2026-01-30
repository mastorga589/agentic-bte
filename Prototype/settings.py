from dataclasses import dataclass
import os

# Auto-load .env in the Prototype directory so tests/use outside Agent still get keys
try:
    from dotenv import load_dotenv
    _ = load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    pass

@dataclass
class Settings:
    # OpenAI
    openai_api_key: str | None = None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    openai_max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))

    # BTE (BioThings Explorer)
    bte_api_base_url: str = os.getenv("BTE_API_BASE_URL", "https://bte.transltr.io/v1")
    bte_query_endpoint: str = os.getenv("BTE_QUERY_ENDPOINT", "asyncquery")
    bte_meta_kg_endpoint: str = os.getenv("BTE_META_KG_ENDPOINT", "meta_knowledge_graph")
    trapi_batch_limit: int = int(os.getenv("TRAPI_BATCH_LIMIT", "10"))

    # Predicate / Meta-KG
    excluded_predicates: list[str] = None

    # Debug
    debug_output_dir: str = os.getenv("PROTOTYPE_DEBUG_DIR", "debug_output")


def get_settings() -> Settings:
    key = os.getenv("OPENAI_API_KEY")
    excluded = [
        # Keep this list light; adjust as needed for your environment
    ]
    s = Settings(
        openai_api_key=key,
        excluded_predicates=excluded,
    )
    # Ensure debug directory exists
    try:
        os.makedirs(s.debug_output_dir, exist_ok=True)
    except Exception:
        pass
    return s
