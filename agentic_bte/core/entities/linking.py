"""
Entity linking for biomedical NER—core component only
"""
import logging
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from ...config.settings import get_settings
from ...exceptions.entity_errors import EntityLinkingError
from agentic_bte.core.entities.recognition import BiomedicalEntityRecognizer

logger = logging.getLogger(__name__)

class EntityLinker:
    """Robust entity linking for biomedical entities (UMLS, SRI)"""
    def __init__(self, openai_api_key: Optional[str] = None):
        self.settings = get_settings()
        self.openai_api_key = openai_api_key or self.settings.openai_api_key
        if not self.openai_api_key:
            raise EntityLinkingError("OpenAI API key is required for entity linking")
        self.llm = ChatOpenAI(
            temperature=0,
            model=self.settings.openai_model,
            api_key=self.openai_api_key
        )
        self.recognizer = BiomedicalEntityRecognizer(openai_api_key)

    def link_entities(self, entity_list: List[str], query: str) -> Dict[str, Dict[str, str]]:
        entity_data = {}
        for entity in entity_list:
            if not entity or entity.strip() == "":
                continue
            try:
                entity_type = self.recognizer.classify_entity(query, entity)
                entity_id = self._link_single_entity(entity, entity_type, query)
                if entity_id:
                    entity_data[entity] = {"id": entity_id, "type": entity_type}
                    logger.debug(f"Linked '{entity}' -> {entity_id} ({entity_type})")
                else:
                    logger.warning(f"Could not link entity: {entity}")
            except Exception as e:
                logger.error(f"Error linking entity '{entity}': {e}")
                continue
        logger.info(f"Successfully linked {len(entity_data)} entities")
        return entity_data

    def _link_single_entity(self, entity: str, entity_type: str, query: str) -> str:
        if entity_type == "biologicalProcess":
            return self._link_biological_process(entity, query)
        else:
            umls_id = self._link_with_umls(entity, query)
            if umls_id:
                return umls_id
            return self._link_with_sri_resolver(entity, query, is_bp=False)

    # ... (include only actively called linking logic)
