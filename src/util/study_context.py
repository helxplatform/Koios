"""
Unified Study Context Provider

Fetches study metadata from FalkorDB for use in both QV and KG chains.
Replaces the need for 99_studies.json file.
"""

from functools import lru_cache
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import html

import config
from databases.redis_graph import RedisGraphDB


@dataclass
class StudyContext:
    """Data class representing complete study context"""
    study_id: str
    study_name: str
    short_name: str
    study_design: str
    participant_count: int
    permalink: str
    abstract: str
    consent_groups: List[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_xml(self, include_variables: bool = False, variables: List[Dict] = None) -> str:
        """Convert study context to XML format for LLM"""
        safe_abstract = html.escape(self.abstract) if self.abstract else ""

        xml = f'''<study id="{self.study_id}">
  <title>{html.escape(self.study_name)} ({self.short_name})</title>
  <study_design>{html.escape(self.study_design or "")}</study_design>
  <participant_count>{self.participant_count}</participant_count>
  <permalink>{self.permalink}</permalink>
  <abstract>{safe_abstract}</abstract>'''

        if include_variables and variables:
            xml += "\n  <variables>"
            for var in variables:
                var_id = var.get('id', '')
                var_name = html.escape(var.get('name', ''))
                var_desc = html.escape(var.get('description', ''))
                xml += f'\n    <variable id="{var_id}">{var_name}: {var_desc}</variable>'
            xml += "\n  </variables>"

        xml += "\n</study>"
        return xml


class StudyContextProvider:
    """
    Provides study context from FalkorDB.

    Usage:
        provider = StudyContextProvider()

        # Single study
        context = provider.get_study("phs000007")

        # Multiple studies (batch)
        contexts = provider.get_studies(["phs000007", "phs000166"])

        # Get as XML for LLM context
        xml = provider.get_study_xml("phs000007")
    """

    _instance = None
    _cache: Dict[str, StudyContext] = {}

    def __new__(cls, *args, **kwargs):
        """Singleton pattern - reuse the same instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, graph_config=None):
        if self._initialized:
            return
        self._config = graph_config or config
        self._graph = RedisGraphDB(self._config)
        self._cache = {}
        self._initialized = True

    def _normalize_study_id(self, study_id: str) -> str:
        """Extract base study ID (e.g., phs000007 from phs000007.v31.p12)"""
        return study_id.split('.')[0] if study_id else ""

    def get_study(self, study_id: str, use_cache: bool = True) -> Optional[StudyContext]:
        """
        Get study context for a single study ID.

        Args:
            study_id: Study ID (e.g., "phs000007" or "phs000007.v31.p12")
            use_cache: Whether to use cached results

        Returns:
            StudyContext object or None if not found
        """
        base_id = self._normalize_study_id(study_id)

        # Check cache
        if use_cache and base_id in self._cache:
            return self._cache[base_id]

        # Query FalkorDB - prioritize studies with abstract and participant data
        result = self._graph.query_graph_sync(f"""
            MATCH (s:`biolink.Study`)
            WHERE s.id STARTS WITH '{base_id}'
            OPTIONAL MATCH (s)-[:`biolink.related_to`]->(p:`biolink.StudyPopulation`)
            WITH s, collect(DISTINCT p.num_subjects) as participant_counts,
                 collect(DISTINCT p.consent_text) as consent_groups
            RETURN
                s.id,
                s.name,
                s.short_study_name,
                s.study_design,
                s.url,
                s.abstract,
                participant_counts,
                consent_groups
            ORDER BY s.abstract IS NOT NULL DESC, size(participant_counts) DESC
            LIMIT 1
        """)

        if not result.result_set:
            return None

        row = result.result_set[0]

        # Sum participant counts (may have multiple consent groups)
        participant_counts = [int(c) for c in row[6] if c and c.isdigit()]
        total_participants = sum(participant_counts) if participant_counts else 0

        context = StudyContext(
            study_id=row[0] or study_id,
            study_name=row[1] or "",
            short_name=row[2] or "",
            study_design=row[3] or "",
            permalink=row[4] or "",
            abstract=row[5] or "",
            participant_count=total_participants,
            consent_groups=row[7] if row[7] else []
        )

        # Cache result
        self._cache[base_id] = context
        return context

    def get_studies(self, study_ids: List[str], use_cache: bool = True) -> Dict[str, StudyContext]:
        """
        Get study context for multiple study IDs (batch operation).

        Args:
            study_ids: List of study IDs
            use_cache: Whether to use cached results

        Returns:
            Dict mapping study_id to StudyContext
        """
        results = {}
        uncached_ids = []

        # Check cache first
        for sid in study_ids:
            base_id = self._normalize_study_id(sid)
            if use_cache and base_id in self._cache:
                results[base_id] = self._cache[base_id]
            else:
                uncached_ids.append(base_id)

        if not uncached_ids:
            return results

        # Batch query for uncached IDs
        id_conditions = " OR ".join([f"s.id STARTS WITH '{sid}'" for sid in set(uncached_ids)])

        result = self._graph.query_graph_sync(f"""
            MATCH (s:`biolink.Study`)
            WHERE {id_conditions}
            OPTIONAL MATCH (s)-[:`biolink.related_to`]->(p:`biolink.StudyPopulation`)
            RETURN
                s.id,
                s.name,
                s.short_study_name,
                s.study_design,
                s.url,
                s.abstract,
                collect(DISTINCT p.num_subjects) as participant_counts,
                collect(DISTINCT p.consent_text) as consent_groups
        """)

        for row in result.result_set:
            study_id = row[0]
            base_id = self._normalize_study_id(study_id)

            participant_counts = [int(c) for c in row[6] if c and c.isdigit()]
            total_participants = sum(participant_counts) if participant_counts else 0

            context = StudyContext(
                study_id=study_id,
                study_name=row[1] or "",
                short_name=row[2] or "",
                study_design=row[3] or "",
                permalink=row[4] or "",
                abstract=row[5] or "",
                participant_count=total_participants,
                consent_groups=row[7] if row[7] else []
            )

            self._cache[base_id] = context
            results[base_id] = context

        return results

    def get_study_xml(self, study_id: str, variables: List[Dict] = None) -> str:
        """
        Get study context as XML string for LLM.

        Args:
            study_id: Study ID
            variables: Optional list of variable dicts with 'id', 'name', 'description'

        Returns:
            XML string or empty study tag if not found
        """
        context = self.get_study(study_id)
        if not context:
            return f'<study id="{study_id}"><error>Study not found</error></study>'

        return context.to_xml(include_variables=bool(variables), variables=variables)

    def get_studies_xml(self, study_ids: List[str], variables_map: Dict[str, List[Dict]] = None) -> str:
        """
        Get multiple studies as XML string for LLM.

        Args:
            study_ids: List of study IDs
            variables_map: Optional dict mapping study_id to list of variables

        Returns:
            XML string wrapped in <studies> tag
        """
        contexts = self.get_studies(study_ids)
        variables_map = variables_map or {}

        xml_parts = []
        for base_id, context in contexts.items():
            variables = variables_map.get(base_id, [])
            xml_parts.append(context.to_xml(include_variables=bool(variables), variables=variables))

        return "<studies>\n" + "\n".join(xml_parts) + "\n</studies>"

    def clear_cache(self):
        """Clear the study context cache"""
        self._cache.clear()


# Convenience function for quick access
def get_study_context(study_id: str) -> Optional[StudyContext]:
    """Quick access to get a single study context"""
    return StudyContextProvider().get_study(study_id)


def get_studies_context(study_ids: List[str]) -> Dict[str, StudyContext]:
    """Quick access to get multiple study contexts"""
    return StudyContextProvider().get_studies(study_ids)


# For backwards compatibility with study_data.py
def get_study_data(study_id: str, comparator=None, exclude_keys=None):
    """
    Backwards compatible function replacing the JSON-based lookup.

    Returns:
        Tuple of (data_dict, status_code)
    """
    context = get_study_context(study_id)

    if not context:
        default_data = {"study_name": "", "permalink": "", "description": "", "study_id": ""}
        if exclude_keys:
            default_data = {k: v for k, v in default_data.items() if k not in exclude_keys}
        return default_data, 404

    data = {
        "study_id": context.study_id,
        "study_name": context.study_name,
        "permalink": context.permalink,
        "description": context.abstract
    }

    if exclude_keys:
        data = {k: v for k, v in data.items() if k not in exclude_keys}

    return data, 200


if __name__ == "__main__":
    # Test the provider
    provider = StudyContextProvider()

    # Test single study
    print("=== Single Study ===")
    context = provider.get_study("phs000007")
    if context:
        print(f"Study: {context.study_name}")
        print(f"Participants: {context.participant_count}")
        print(f"Design: {context.study_design}")

    # Test XML output
    print("\n=== XML Output ===")
    xml = provider.get_study_xml("phs000007", variables=[
        {"id": "phv00021388", "name": "diabetes_age", "description": "Age at diabetes diagnosis"}
    ])
    print(xml)

    # Test batch
    print("\n=== Batch Query ===")
    contexts = provider.get_studies(["phs000007", "phs000166", "phs000179"])
    for sid, ctx in contexts.items():
        print(f"{ctx.study_name}: {ctx.participant_count} participants")
