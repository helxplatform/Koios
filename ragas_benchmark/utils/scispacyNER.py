import spacy
import typing as t
from ragas.testset.transforms.base import Node, Extractor  # adjust to your actual Node import
from ragas.testset.transforms.base import default_filter, KnowledgeGraph, field

class SciSpacyNERExtractor(Extractor):
    """
    Extracts named entities from text using SciSpacy instead of an LLM.

    Attributes
    ----------
    property_name : str
        The name of the property to extract. Defaults to "entities".
    model_name : str
        The SciSpacy model to load (default: en_core_sci_sm).
    """

    property_name: str = "entities"

    def __init__(self, model_name: str = "en_core_sci_sm"):
        self.model_name = model_name
        self.nlp = spacy.load(model_name)

    async def extract(self, node: Node, *args, **kwargs ) -> t.Tuple[str, t.List[str]]:
        """
        Extract entities from the node's text content.
        """
        node_text = node.get_property("page_content")
        if not node_text:
            return self.property_name, []

        doc = self.nlp(node_text)
        entities = list(set([ent.text for ent in doc.ents]))

        return self.property_name, entities



    def filter(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        """
        Filters the KnowledgeGraph and returns the filtered graph.

        Parameters
        ----------
        kg : KnowledgeGraph
            The knowledge graph to be filtered.

        Returns
        -------
        KnowledgeGraph
            The filtered knowledge graph.
        """

        return KnowledgeGraph(
            nodes=[node for node in kg.nodes if default_filter(node)],
            relationships=[
                rel
                for rel in kg.relationships
                if rel.source in kg.nodes and rel.target in kg.nodes
            ],
        )