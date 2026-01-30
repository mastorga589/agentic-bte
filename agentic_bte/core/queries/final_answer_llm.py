"""
LLM-Based Final Answer Generation for RAG System

This module provides LLM-powered final answer synthesis that leverages
all collected biomedical evidence, relationships, and entity mappings
to generate comprehensive, research-quality responses.
"""

import logging
import json
import asyncio
import requests
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AnswerContext:
    """Context information for generating final answers"""
    query: str
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    entity_mappings: Dict[str, str]
    execution_metadata: Dict[str, Any]
    subquery_execution_plan: List[Dict[str, Any]] = None
    subquery_results: Dict[str, Any] = None
    confidence_threshold: float = 0.3


class LLMFinalAnswerGenerator:
    """
    LLM-powered final answer generator for biomedical queries
    
    Synthesizes comprehensive responses using all available evidence
    from knowledge graph exploration and parallel predicate execution.
    """
    
    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.1):
        """
        Initialize the LLM answer generator
        
        Args:
            model_name: OpenAI model to use for generation (defaults to settings.openai_model)
            temperature: Temperature setting for generation
        """
        # Default to configured model if not explicitly provided
        try:
            from ...config.settings import get_settings
            self.model_name = model_name or get_settings().openai_model
        except Exception:
            self.model_name = model_name or "gpt-4o"
        self.temperature = temperature
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM client"""
        try:
            from ...config.settings import get_settings
            from langchain_openai import ChatOpenAI
            
            settings = get_settings()
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=settings.openai_api_key,
                max_tokens=2000
            )
            logger.info(f"Initialized LLM finalizer with {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise RuntimeError(f"LLM initialization failed: {e}")
    
    async def generate_final_answer(self, query: str, final_results: List[Dict[str, Any]], 
                                  entities: List[Dict[str, Any]], 
                                  execution_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate comprehensive final answer using LLM synthesis
        
        Args:
            query: Original biomedical query
            final_results: Ranked and deduplicated results from knowledge graph
            entities: Extracted biomedical entities
            execution_context: Additional execution metadata
            
        Returns:
            LLM-generated comprehensive final answer
        """
        if not final_results:
            return self._generate_no_results_answer(query)
        
        try:
            # Prepare structured context
            context = await self._prepare_answer_context(
                query, final_results, entities, execution_context or {}
            )
            
            # Build comprehensive prompt
            prompt = self._build_comprehensive_prompt(context)
            
            # Log prompt for auditing/debugging (trimmed)
            try:
                preview = (prompt[:1000] + "...") if len(prompt) > 1000 else prompt
                logger.debug(f"LLM Final Answer Prompt Preview:\n{preview}")
            except Exception:
                pass
            
            # Generate answer using LLM
            logger.info(f"Generating LLM final answer for query: {query[:100]}...")
            # Use synchronous invoke in a thread to avoid async httpx client lifecycle issues
            response = await asyncio.to_thread(self.llm.invoke, [{"role": "user", "content": prompt}])
            
            final_answer = (response.content if hasattr(response, "content") else str(response)).strip()
            logger.info(f"Generated LLM final answer ({len(final_answer)} characters)")
            
            return final_answer
            
        except Exception as e:
            logger.error(f"LLM final answer generation failed: {e}")
            return self._generate_fallback_answer(query, final_results, entities)
    
    async def _batch_resolve_entity_names(self, entity_ids: set) -> Dict[str, str]:
        """
        Batch resolve entity IDs to human-readable names using multiple APIs
        
        Args:
            entity_ids: Set of entity IDs to resolve
            
        Returns:
            Dictionary mapping entity_id -> human_readable_name
        """
        resolved_names = {}
        
        try:
            async def resolve_single_id(entity_id: str) -> Optional[Tuple[str, str]]:
                """Resolve a single entity ID using multiple strategies (synchronously in threads)."""
                try:
                    # Strategy 1: MyGene.info for gene IDs
                    if 'NCBIGene:' in entity_id or 'HGNC:' in entity_id or 'ENSEMBL:' in entity_id:
                        gene_id = entity_id.split(':')[-1]
                        def _req_gene():
                            return requests.get(f"https://mygene.info/v3/gene/{gene_id}", timeout=10)
                        resp = await asyncio.to_thread(_req_gene)
                        if resp.status_code == 200:
                            data = resp.json()
                            name = data.get('name') or data.get('symbol')
                            if name:
                                return entity_id, name
                    # Strategy 2: MyChemical.info for chemical IDs
                    elif 'CHEBI:' in entity_id or 'ChEMBL:' in entity_id or 'DRUGBANK:' in entity_id:
                        chem_id = entity_id.replace(':', '%3A')
                        def _req_chem():
                            return requests.get(f"https://mychem.info/v1/chem/{chem_id}", timeout=10)
                        resp = await asyncio.to_thread(_req_chem)
                        if resp.status_code == 200:
                            data = resp.json()
                            name = data.get('_id') or (data.get('chebi', {}) if isinstance(data.get('chebi', {}), dict) else {}).get('name')
                            if name:
                                return entity_id, name
                    # Strategy 3: OLS for GO/HP
                    elif 'GO:' in entity_id or 'HP:' in entity_id:
                        def _req_ols():
                            return requests.get(
                                f"https://www.ebi.ac.uk/ols/api/terms?iri=http://purl.obolibrary.org/obo/{entity_id.replace(':', '_')}",
                                timeout=10,
                            )
                        resp = await asyncio.to_thread(_req_ols)
                        if resp.status_code == 200:
                            data = resp.json()
                            terms = data.get('_embedded', {}).get('terms', [])
                            if terms:
                                name = terms[0].get('label')
                                if name:
                                    return entity_id, name
                    # Strategy 4: simple heuristics
                    if ':' in entity_id:
                        prefix, suffix = entity_id.split(':', 1)
                        if prefix in ['MONDO', 'UMLS', 'DOID']:
                            return entity_id, f"{prefix} {suffix}"
                        if prefix == 'CHEBI':
                            return entity_id, f"Chemical {suffix}"
                        if prefix == 'NCBIGene':
                            return entity_id, f"Gene {suffix}"
                    return None
                except Exception as e:
                    logger.debug(f"Failed to resolve {entity_id}: {e}")
                    return None
            # Resolve concurrently with a small semaphore
            semaphore = asyncio.Semaphore(5)
            async def resolve_with_semaphore(entity_id: str):
                async with semaphore:
                    return await resolve_single_id(entity_id)
            resolution_tasks = [resolve_with_semaphore(entity_id) for entity_id in entity_ids]
            results = await asyncio.gather(*resolution_tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, tuple) and len(result) == 2:
                    entity_id, name = result
                    resolved_names[entity_id] = name
                elif isinstance(result, Exception):
                    logger.debug(f"Resolution task failed: {result}")
            
            logger.info(f"Successfully resolved {len(resolved_names)} out of {len(entity_ids)} entity names")
            
        except Exception as e:
            logger.error(f"Batch entity resolution failed: {e}")
        
        return resolved_names
    
    async def _prepare_answer_context(self, query: str, final_results: List[Dict[str, Any]], 
                              entities: List[Dict[str, Any]], 
                              execution_context: Dict[str, Any]) -> AnswerContext:
        """Prepare structured context for LLM prompt"""
        
        # Extract entity mappings from results and entities
        entity_mappings = {}
        
        # First, add mappings from the original entities list
        for entity in entities:
            name = entity.get('name', '')
            entity_id = entity.get('id', '')
            if name and entity_id:
                entity_mappings[entity_id] = name
                entity_mappings[name] = entity_id
        
        # CRITICAL FIX: Extract entity names from knowledge graph nodes in final_results
        # This resolves UMLS IDs and other entity IDs to human-readable names
        logger.debug(f"Extracting entity names from {len(final_results)} BTE results...")
        kg_entities_resolved = 0
        unresolved_ids = set()
        
        for result in final_results:
            kg = result.get('knowledge_graph', {})
            if not kg:
                continue
                
            nodes = kg.get('nodes', {})
            for node_id, node_data in nodes.items():
                name = node_data.get('name')
                if name and node_id and node_id not in entity_mappings:
                    entity_mappings[node_id] = name
                    entity_mappings[name] = node_id
                    kg_entities_resolved += 1
                elif node_id and not name and node_id not in entity_mappings:
                    # Collect unresolved IDs for batch resolution
                    unresolved_ids.add(node_id)
        
        # Batch resolve unresolved entity IDs using multiple APIs
        if unresolved_ids:
            logger.info(f"Attempting to resolve {len(unresolved_ids)} unresolved entity IDs...")
            additional_resolved = await self._batch_resolve_entity_names(unresolved_ids)
            entity_mappings.update(additional_resolved)
            kg_entities_resolved += len(additional_resolved)
        
        logger.info(f"Resolved {kg_entities_resolved} additional entity names from knowledge graph nodes")
        logger.debug(f"Total entity mappings: {len(entity_mappings)}")
        
        # Process relationships from final_results
        relationships = []
        for result in final_results[:50]:  # Top 50 results
            # Extract relationship information
            relationship = self._extract_relationship_from_result(result, entity_mappings)
            if relationship:
                relationships.append(relationship)
        
        # Extract subquery execution plan and results from execution context
        subquery_plan, subquery_results = self._extract_subquery_information(execution_context, final_results)
        
        return AnswerContext(
            query=query,
            entities=entities,
            relationships=relationships,
            entity_mappings=entity_mappings,
            execution_metadata=execution_context,
            subquery_execution_plan=subquery_plan,
            subquery_results=subquery_results
        )
    
    def _extract_relationship_from_result(self, result: Dict[str, Any], 
                                        entity_mappings: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Extract structured relationship information from a result"""
        try:
            # Try to extract from knowledge graph structure
            kg = result.get('knowledge_graph', {})
            if kg and isinstance(kg, dict):
                nodes = kg.get('nodes', {})
                edges = kg.get('edges', {})
                
                if edges and nodes:
                    # Get first edge for this relationship
                    edge_id, edge_data = next(iter(edges.items()))
                    subject_id = edge_data.get('subject')
                    object_id = edge_data.get('object')
                    predicates = edge_data.get('predicates', [])
                    
                    if subject_id in nodes and object_id in nodes:
                        subject_name = nodes[subject_id].get('name', subject_id)
                        object_name = nodes[object_id].get('name', object_id)
                        predicate = predicates[0] if predicates else 'related_to'
                        
                        return {
                            'subject': subject_name,
                            'predicate': predicate,
                            'object': object_name,
                            'confidence': result.get('score', 0.0),
                            'evidence_attributes': edge_data.get('attributes', []),
                            'source_databases': self._extract_sources(edge_data)
                        }
            
            # Fallback: extract from direct result structure
            if 'subject_id' in result and 'object_id' in result:
                subject_name = entity_mappings.get(result['subject_id'], result['subject_id'])
                object_name = entity_mappings.get(result['object_id'], result['object_id'])
                predicate = result.get('predicate', 'related_to')
                
                return {
                    'subject': subject_name,
                    'predicate': predicate,
                    'object': object_name,
                    'confidence': result.get('score', 0.0),
                    'evidence_attributes': result.get('attributes', []),
                    'source_databases': result.get('sources', [])
                }
                
        except Exception as e:
            logger.debug(f"Failed to extract relationship from result: {e}")
        
        return None
    
    def _extract_sources(self, edge_data: Dict[str, Any]) -> List[str]:
        """Extract source database information from edge attributes"""
        sources = []
        attributes = edge_data.get('attributes', [])
        
        for attr in attributes:
            if isinstance(attr, dict):
                attr_type = attr.get('attribute_type_id', '')
                value = attr.get('value', '')
                
                if 'source' in attr_type.lower() and value:
                    sources.append(value)
                elif 'provided_by' in attr_type.lower() and value:
                    sources.append(value)
        
        return sources[:3]  # Limit to top 3 sources
    
    def _extract_subquery_information(self, execution_context: Dict[str, Any], 
                                    final_results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Extract subquery execution plan and results from execution context"""
        subquery_plan = []
        subquery_results = {}
        
        try:
            execution_steps = execution_context.get('execution_steps', [])
            subquery_info_list = execution_context.get('subquery_info', [])
            
            # If we have actual subquery information from the optimizer, use it
            if subquery_info_list:
                subquery_plan = []
                for i, sq_info in enumerate(subquery_info_list):
                    entities_used = [e.get('name', 'Unknown') for e in sq_info.get('entities', [])]
                    
                    # Count results for this subquery by matching step patterns
                    results_found = 0
                    top_confidence = 0.0
                    execution_success = False
                    
                    # Look for matching API execution steps
                    for step in execution_steps:
                        if (step.get('step_type') == 'api_execution' and 
                            f'subq_{i}' in step.get('step_id', '')):
                            if step.get('success'):
                                execution_success = True
                                step_results = step.get('output_data', {}).get('results', [])
                                results_found += len(step_results)
                                top_confidence = max(top_confidence, step.get('confidence', 0.0))
                    
                    # Consider a subquery failed if it produced zero results overall
                    execution_success = bool(execution_success and results_found > 0)
                    subquery_plan.append({
                        'subquery_number': i + 1,
                        'query_text': sq_info.get('query', f'Subquery {i + 1}'),
                        'entities_used': entities_used,
                        'rationale': sq_info.get('rationale', 'Scientific decomposition step'),
                        'execution_success': execution_success,
                        'results_found': results_found,
                        'top_confidence': top_confidence
                    })
                
                # Extract top results for each subquery concept
                subquery_results = self._group_results_by_subquery_concepts(final_results, subquery_plan)
                return subquery_plan, subquery_results
            
            # Fallback: Group steps by subquery based on step_id patterns
            current_subquery = None
            subquery_counter = 1
            
            for step in execution_steps:
                step_type = step.get('step_type', '')
                step_id = step.get('step_id', '')
                success = step.get('success', False)
                confidence = step.get('confidence', 0.0)
                
                # Detect when we start a new subquery (TRAPI query building step)
                if step_type == 'trapi_query_building' and 'subq_' in step_id:
                    # Extract subquery info from step metadata if available
                    if current_subquery:
                        subquery_plan.append(current_subquery)
                    
                    current_subquery = {
                        'subquery_number': subquery_counter,
                        'query_text': f"Subquery {subquery_counter}",  # Will be enhanced if available
                        'entities_used': [],
                        'rationale': f"Scientific decomposition step {subquery_counter}",
                        'execution_success': success,
                        'results_found': 0,
                        'top_confidence': confidence
                    }
                    subquery_counter += 1
                
                # Track API execution results for each subquery
                elif step_type == 'api_execution' and current_subquery:
                    if success and 'output_data' in step:
                        results_count = len(step.get('output_data', {}).get('results', []))
                        current_subquery['results_found'] += results_count
                        current_subquery['top_confidence'] = max(current_subquery['top_confidence'], confidence)
            
            # Add the last subquery
            if current_subquery:
                subquery_plan.append(current_subquery)
            
            # If no subqueries detected, create a default plan
            if not subquery_plan:
                api_steps = [s for s in execution_steps if s.get('step_type') == 'api_execution' and s.get('success')]
                if api_steps:
                    subquery_plan.append({
                        'subquery_number': 1,
                        'query_text': 'Single comprehensive query execution',
                        'entities_used': [e.get('name', 'Unknown') for e in execution_context.get('entities', [])[:5]],
                        'rationale': 'Direct knowledge graph exploration of the original query',
                        'execution_success': True,
                        'results_found': len(final_results),
                        'top_confidence': max((s.get('confidence', 0.0) for s in api_steps), default=0.0)
                    })
            
            # Extract top results for each subquery concept
            subquery_results = self._group_results_by_subquery_concepts(final_results, subquery_plan)
            
        except Exception as e:
            logger.debug(f"Failed to extract subquery information: {e}")
            # Fallback: create basic execution summary
            subquery_plan = [{
                'subquery_number': 1,
                'query_text': 'Knowledge graph exploration',
                'entities_used': [e.get('name', 'Unknown') for e in execution_context.get('entities', [])[:3]],
                'rationale': 'Comprehensive biomedical knowledge search',
                'execution_success': True,
                'results_found': len(final_results),
                'top_confidence': 0.0
            }]
            subquery_results = {'general': final_results[:10]}
        
        return subquery_plan, subquery_results
    
    def _group_results_by_subquery_concepts(self, final_results: List[Dict[str, Any]], 
                                          subquery_plan: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group results by originating subquery (using annotated source_subquery)."""
        grouped_results: Dict[str, List[Dict[str, Any]]] = {}
        
        # Index results by annotated source_subquery
        results_by_src: Dict[int, List[Dict[str, Any]]] = {}
        for r in final_results:
            idx = r.get('source_subquery')
            if isinstance(idx, int):
                results_by_src.setdefault(idx, []).append(r)
        
        # For each subquery in the plan, collect top results truly from that subquery
        for i, subquery in enumerate(subquery_plan):
            subq_num = i + 1
            subquery_key = f"subquery_{subq_num}"
            candidates = results_by_src.get(subq_num, [])
            # Sort by score desc if available
            candidates = sorted(candidates, key=lambda x: x.get('score', 0.0), reverse=True)
            grouped_results[subquery_key] = candidates[:5]  # show up to 5
        
        return grouped_results
    
    def _build_comprehensive_prompt(self, context: AnswerContext) -> str:
        """Build comprehensive prompt for LLM final answer generation"""
        
        # Classify query type for specialized instructions
        query_type = self._classify_query_type(context.query)
        
        # Build entity summary
        entity_summary = self._build_entity_summary(context.entities, context.entity_mappings)
        
        # Build relationship summary
        relationship_summary = self._build_relationship_summary(context.relationships, context.entity_mappings)
        
        # Build execution plan summary
        execution_plan_summary = self._build_execution_plan_summary(context.subquery_execution_plan)
        
        # Build subquery results showcase
        subquery_results_showcase = self._build_subquery_results_showcase(context.subquery_results)
        
        # Quality assessment
        quality_metrics = self._assess_data_quality(context)
        
        prompt = f"""You are an expert biomedical researcher analyzing data from comprehensive knowledge graphs including FDA databases, clinical trials, genomic studies, and peer-reviewed literature.

**QUERY TO ANSWER:**
{context.query}

**ANALYSIS OBJECTIVE:**
Generate a comprehensive, scientifically accurate response based on the evidence below. This is a RAG (Retrieval-Augmented Generation) system, so your answer must be grounded in the provided evidence while applying your biomedical expertise for context and interpretation.

**BIOMEDICAL ENTITIES IDENTIFIED:**
{entity_summary}

**QUERY EXECUTION PLAN:**
{execution_plan_summary}

**KEY EVIDENCE BY SUBQUERY:**
{subquery_results_showcase}

**COMPREHENSIVE EVIDENCE RELATIONSHIPS ({len(context.relationships)} total):**
{relationship_summary}

**DATA QUALITY ASSESSMENT:**
{quality_metrics}

**RESPONSE REQUIREMENTS - YOU MUST INCLUDE THESE SECTIONS:**

1. **Primary Answer**: Start with a direct, clear answer to the user's question

2. **QUERY EXECUTION PLAN:** (REQUIRED SECTION)
   Copy and adapt the execution plan information provided above. Include:
   - Brief summary of the systematic subquery approach used
   - List the specific subqueries that were executed
   - Show the success status and results found for each subquery

3. **KEY EVIDENCE BY SUBQUERY:** (REQUIRED SECTION)
   Copy and adapt the key evidence breakdown provided above. Include:
   - Evidence discovered from each subquery
   - Specific relationships with confidence scores
   - Entity names and relationship types

4. **Evidence-Based Analysis**: 
   - Reference specific relationships and confidence scores from the data
   - Explain the biological/clinical significance of key findings
   - Connect entities through mechanisms, pathways, or clinical associations

5. **Scientific Context**: 
   - Apply your expertise to interpret findings in broader biomedical context
   - Explain mechanisms of action, therapeutic rationale, or genetic associations
   - Note any limitations or gaps in the current evidence

6. **Quality Transparency**: 
   - Be clear about evidence strength (high/medium/low confidence)
   - Distinguish between well-established vs. emerging evidence
   - Recommend further research if evidence is incomplete

**SPECIFIC INSTRUCTIONS FOR THIS QUERY TYPE ({query_type}):**
{self._get_query_specific_instructions(query_type)}

**FORMATTING REQUIREMENTS:**
- Start your response with the Primary Answer section
- If the question asks for drugs/therapeutics, begin the Primary Answer with exactly 5 lines, each containing only one drug name formatted as **Drug Name** (no numbering, no extra text on those five lines). After those five lines, continue with the rest of the sections.
- MUST include a "**QUERY EXECUTION PLAN:**" section with the execution information
- MUST include a "**KEY EVIDENCE BY SUBQUERY:**" section with the evidence breakdown
- Use clear markdown headings (##, ###) and bullet points for readability
- Include confidence indicators (e.g., "strongly supported", "preliminary evidence")
- Cite key relationships by their confidence scores when relevant
- Keep the response comprehensive but focused (aim for 500-800 words total)

Generate your evidence-based response now, ensuring you include the required QUERY EXECUTION PLAN and KEY EVIDENCE BY SUBQUERY sections:"""

        return prompt
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type for specialized instructions"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['drug', 'treat', 'medicine', 'therapeutic']):
            return 'therapeutic'
        elif any(word in query_lower for word in ['gene', 'genetic', 'protein']):
            return 'genetic'
        elif any(word in query_lower for word in ['mechanism', 'pathway', 'how', 'why']):
            return 'mechanistic'
        elif any(word in query_lower for word in ['disease', 'disorder', 'condition']):
            return 'disease_focused'
        else:
            return 'general_biomedical'
    
    def _get_query_specific_instructions(self, query_type: str) -> str:
        """Get query-type specific instructions"""
        instructions = {
            'therapeutic': "Focus on therapeutic compounds, their mechanisms of action, clinical efficacy, FDA approval status, and contraindications. Prioritize drugs with highest confidence scores.",
            'genetic': "Emphasize genetic associations, protein functions, inheritance patterns, and clinical relevance. Include information about genetic testing and personalized medicine implications.",
            'mechanistic': "Detail molecular mechanisms, signaling pathways, protein interactions, and biological processes. Explain how different components work together at the cellular/molecular level.",
            'disease_focused': "Cover disease pathophysiology, clinical manifestations, diagnostic criteria, and associated risk factors. Connect genetic, environmental, and therapeutic aspects.",
            'general_biomedical': "Provide a balanced overview covering relevant entities, relationships, and biological significance. Structure around the most confident evidence."
        }
        
        return instructions.get(query_type, instructions['general_biomedical'])
    
    def _build_entity_summary(self, entities: List[Dict[str, Any]], 
                            entity_mappings: Dict[str, str]) -> str:
        """Build structured entity summary"""
        if not entities:
            return "No specific entities identified."
        
        # Group entities by type
        entity_groups = {
            'Diseases/Conditions': [],
            'Drugs/Chemicals': [],
            'Genes/Proteins': [],
            'Other Biological Entities': []
        }
        
        for entity in entities[:20]:  # Top 20 entities
            name = entity.get('name', 'Unknown')
            entity_type = entity.get('type', '').lower()
            entity_id = entity.get('id', '')
            confidence = entity.get('confidence', 0.0)
            
            # Classify entity
            if any(term in entity_type for term in ['disease', 'disorder', 'condition']):
                entity_groups['Diseases/Conditions'].append(f"• {name} (ID: {entity_id}, conf: {confidence:.2f})")
            elif any(term in entity_type for term in ['chemical', 'drug', 'compound']):
                entity_groups['Drugs/Chemicals'].append(f"• {name} (ID: {entity_id}, conf: {confidence:.2f})")
            elif any(term in entity_type for term in ['gene', 'protein']):
                entity_groups['Genes/Proteins'].append(f"• {name} (ID: {entity_id}, conf: {confidence:.2f})")
            else:
                entity_groups['Other Biological Entities'].append(f"• {name} (ID: {entity_id}, conf: {confidence:.2f})")
        
        summary_parts = []
        for group_name, group_entities in entity_groups.items():
            if group_entities:
                summary_parts.append(f"{group_name}:")
                summary_parts.extend(group_entities[:5])  # Top 5 per group
                if len(group_entities) > 5:
                    summary_parts.append(f"... and {len(group_entities) - 5} more")
                summary_parts.append("")
        
        return "\n".join(summary_parts) if summary_parts else "No categorized entities available."
    
    def _build_relationship_summary(self, relationships: List[Dict[str, Any]], 
                                  entity_mappings: Dict[str, str] = None) -> str:
        """Build structured relationship summary with resolved entity names"""
        if not relationships:
            return "No structured relationships identified."
        
        entity_mappings = entity_mappings or {}
        
        # Sort by confidence
        sorted_relationships = sorted(relationships, key=lambda x: x.get('confidence', 0), reverse=True)
        
        summary_parts = []
        for i, rel in enumerate(sorted_relationships[:15], 1):  # Top 15 relationships
            subject_raw = rel['subject']
            object_raw = rel['object']
            
            # Try to resolve entity names using the mappings
            subject = entity_mappings.get(subject_raw, subject_raw)
            object_name = entity_mappings.get(object_raw, object_raw)
            
            # Truncate if still too long
            subject = subject[:40] + "..." if len(subject) > 40 else subject
            object_name = object_name[:40] + "..." if len(object_name) > 40 else object_name
            
            predicate = rel['predicate'].replace('biolink:', '').replace('_', ' ')
            confidence = rel['confidence']
            
            confidence_label = "HIGH" if confidence > 0.7 else "MED" if confidence > 0.4 else "LOW"
            summary_parts.append(f"{i}. {subject} → {predicate} → {object_name} [{confidence_label}: {confidence:.2f}]")
        
        if len(relationships) > 15:
            summary_parts.append(f"... and {len(relationships) - 15} additional relationships")
        
        return "\n".join(summary_parts)
    
    def _assess_data_quality(self, context: AnswerContext) -> str:
        """Assess and summarize data quality"""
        total_relationships = len(context.relationships)
        high_conf = len([r for r in context.relationships if r.get('confidence', 0) > 0.7])
        med_conf = len([r for r in context.relationships if 0.4 < r.get('confidence', 0) <= 0.7])
        low_conf = len([r for r in context.relationships if 0 < r.get('confidence', 0) <= 0.4])
        
        quality_parts = [
            f"• Total evidence relationships: {total_relationships}",
            f"• High confidence evidence (>0.7): {high_conf} ({100*high_conf/max(1,total_relationships):.0f}%)",
            f"• Medium confidence evidence (0.4-0.7): {med_conf} ({100*med_conf/max(1,total_relationships):.0f}%)",
            f"• Lower confidence evidence (<0.4): {low_conf} ({100*low_conf/max(1,total_relationships):.0f}%)",
            f"• Entities identified: {len(context.entities)}"
        ]
        
        return "\n".join(quality_parts)
    
    def _build_execution_plan_summary(self, subquery_plan: Optional[List[Dict[str, Any]]]) -> str:
        """Build execution plan summary for transparency"""
        if not subquery_plan:
            return "Single-step knowledge graph exploration executed."
        
        plan_parts = []
        plan_parts.append(f"The query was systematically decomposed into {len(subquery_plan)} focused subqueries:")
        plan_parts.append("")
        
        for sq in subquery_plan:
            sq_num = sq.get('subquery_number', 'N/A')
            query_text = sq.get('query_text', 'Unknown query')
            entities = sq.get('entities_used', [])
            rationale = sq.get('rationale', 'No rationale provided')
            results_found = sq.get('results_found', 0)
            success = sq.get('execution_success', False)
            
            status_emoji = "✅" if success else "❌"
            plan_parts.append(f"{status_emoji} **Subquery {sq_num}**: {query_text}")
            if entities:
                entities_str = ', '.join(entities[:4])  # Show up to 4 entities
                if len(entities) > 4:
                    entities_str += f" + {len(entities) - 4} more"
                plan_parts.append(f"   - Entities: {entities_str}")
            plan_parts.append(f"   - Scientific rationale: {rationale}")
            plan_parts.append(f"   - Results found: {results_found}")
            plan_parts.append("")
        
        return "\n".join(plan_parts)
    
    def _build_subquery_results_showcase(self, subquery_results: Optional[Dict[str, Any]]) -> str:
        """Build subquery results showcase for explainability"""
        if not subquery_results:
            return "No detailed subquery results available for breakdown."
        
        showcase_parts = []
        showcase_parts.append("Key evidence discovered in each search direction:")
        showcase_parts.append("")
        
        for subquery_key, results in subquery_results.items():
            if not results:
                continue
                
            subquery_num = subquery_key.split('_')[-1]
            showcase_parts.append(f"**From Subquery {subquery_num}** ({len(results)} key relationships):")
            
            # Show top 3-5 results from this subquery
            for i, result in enumerate(results[:5], 1):
                try:
                    # Try to extract meaningful information from the result
                    subject = self._extract_entity_name(result, 'subject')
                    object_name = self._extract_entity_name(result, 'object')
                    predicate = result.get('predicate', 'related_to').replace('biolink:', '').replace('_', ' ')
                    confidence = result.get('score', 0.0)
                    
                    # Truncate long names
                    subject = subject[:30] + "..." if len(subject) > 30 else subject
                    object_name = object_name[:30] + "..." if len(object_name) > 30 else object_name
                    
                    confidence_label = "HIGH" if confidence > 0.7 else "MED" if confidence > 0.4 else "LOW"
                    showcase_parts.append(f"   {i}. {subject} → {predicate} → {object_name} [{confidence_label}: {confidence:.3f}]")
                    
                except Exception as e:
                    # Fallback for malformed results
                    showcase_parts.append(f"   {i}. [Complex relationship - confidence: {result.get('score', 0.0):.3f}]")
            
            showcase_parts.append("")
        
        return "\n".join(showcase_parts)
    
    def _extract_entity_name(self, result: Dict[str, Any], role: str) -> str:
        """Extract entity name from result for the given role (subject/object) with robust fallbacks"""
        def _looks_generic(name: str) -> bool:
            if not isinstance(name, str):
                return True
            n = name.strip().lower()
            return n in {"unknown", "namedthing", "molecule", "small molecule", "chemical", "biological process", "gene", "protein"}

        def _resolve_via_node_norm(eid: str) -> str:
            try:
                import requests
                url = "https://nodenormalization-sri.renci.org/1.5/get_normalized_nodes"
                r = requests.post(url, json={"curies": [eid], "conflate": True, "description": True}, timeout=8)
                r.raise_for_status()
                data = r.json() or {}
                entry = data.get(eid)
                if isinstance(entry, dict):
                    lbl = ((entry.get('id') or {}).get('label'))
                    if isinstance(lbl, str) and lbl.strip():
                        return lbl
                    for eq in entry.get('equivalent_identifiers', []) or []:
                        lbl = eq.get('label')
                        if isinstance(lbl, str) and lbl.strip():
                            return lbl
            except Exception:
                pass
            return eid

        try:
            # Try knowledge graph structure first
            kg = result.get('knowledge_graph', {})
            if kg and isinstance(kg, dict):
                nodes = kg.get('nodes', {})
                edges = kg.get('edges', {})
                
                if edges and nodes:
                    edge_id, edge_data = next(iter(edges.items()))
                    entity_id = edge_data.get(role)
                    if entity_id in nodes:
                        name = nodes[entity_id].get('name', entity_id)
                        if _looks_generic(name) and isinstance(entity_id, str):
                            # Attempt to resolve a better label on the fly
                            return _resolve_via_node_norm(entity_id)
                        return name
            
            # Fallback to direct result structure
            entity_id_key = f'{role}_id'
            entity_name_key = f'{role}_name'
            
            if entity_name_key in result:
                name = result[entity_name_key]
                return name
            elif entity_id_key in result:
                eid = result[entity_id_key]
                if isinstance(eid, str) and eid:
                    return _resolve_via_node_norm(eid)
                return eid
            else:
                return f"Unknown {role}"
                
        except Exception:
            return f"Unknown {role}"
    
    def _generate_no_results_answer(self, query: str) -> str:
        """Generate answer when no results are found"""
        return f"""I was unable to find specific biomedical evidence to answer your query: "{query}"

This could be due to several factors:
• The query may require more specific biomedical terminology
• The knowledge graph may have limited coverage in this particular area
• The entities or relationships may exist but fall below confidence thresholds

**Recommendations:**
• Try rephrasing with more specific medical or scientific terms
• Consider breaking complex queries into simpler components
• Consult specialized databases or recent literature for emerging topics

**Alternative Approaches:**
• Use standardized biomedical vocabularies (e.g., MeSH, UMLS terms)
• Focus on well-established relationships in major databases
• Consider broader related concepts that might yield results"""
    
    def _generate_fallback_answer(self, query: str, results: List[Dict[str, Any]], 
                                entities: List[Dict[str, Any]]) -> str:
        """Generate fallback answer when LLM fails"""
        logger.warning("Using fallback answer generation due to LLM failure")
        
        from .result_presenter import format_final_answer
        return format_final_answer(results, entities, query)


async def llm_generate_final_answer(query: str, final_results: List[Dict[str, Any]], 
                                  entities: List[Dict[str, Any]], 
                                  execution_context: Optional[Dict[str, Any]] = None) -> str:
    """
    Convenience function for LLM-based final answer generation
    
    Args:
        query: Original biomedical query
        final_results: Ranked and deduplicated results
        entities: Extracted entities
        execution_context: Additional execution metadata
        
    Returns:
        LLM-generated comprehensive final answer
    """
    generator = LLMFinalAnswerGenerator()
    return await generator.generate_final_answer(query, final_results, entities, execution_context)