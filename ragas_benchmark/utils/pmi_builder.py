from collections import Counter
import math
from dataclasses import dataclass
from ragas.testset.transforms.base import RelationshipBuilder, KnowledgeGraph, Relationship, Node

@dataclass
class PMIRelationshipBuilder(RelationshipBuilder):
    property_name: str = "entities"
    new_property_name: str = "pmi"
    threshold: float = 0.0

    # REQUIRED
    def filter_nodes(self, kg):
        return kg.nodes

    # REQUIRED
    def filter(self, kg):
        return kg

    async def transform(self, kg: KnowledgeGraph):
        all_entities = [e for n in kg.nodes for e in (n.get_property(self.property_name) or [])]
        entity_counts = Counter(all_entities)
        total = sum(entity_counts.values())

        pair_counts = Counter()
        for n in kg.nodes:
            ents = list(set(n.get_property(self.property_name) or []))
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    pair_counts[tuple(sorted([ents[i], ents[j]]))] += 1

        relationships = []
        for (a, b), c_ab in pair_counts.items():
            p_a = entity_counts[a] / total
            p_b = entity_counts[b] / total
            p_ab = c_ab / len(kg.nodes)
            pmi = math.log2(p_ab / (p_a * p_b)) if p_ab > 0 else -float("inf")

            if pmi > self.threshold:
                relationships.append(
                    Relationship(
                        source=Node(id=a),
                        target=Node(id=b),
                        type="entity_cooccurrence",
                        properties={self.new_property_name: pmi},
                        bidirectional=True,
                    )
                )
        return relationships
