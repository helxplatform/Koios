from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from ragas.testset.transforms.base import RelationshipBuilder, KnowledgeGraph, Relationship

@dataclass
class WeightedJaccardBuilder(RelationshipBuilder):
    property_name: str = "entities"
    new_property_name: str = "weighted_jaccard"
    threshold: float = 0.3

    # REQUIRED
    def filter_nodes(self, kg):
        return kg.nodes

    # REQUIRED
    def filter(self, kg):
        return kg

    async def transform(self, kg: KnowledgeGraph):
        docs = [" ".join(n.get_property(self.property_name) or []) for n in kg.nodes]

        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w-]+\b")
        X = vectorizer.fit_transform(docs).toarray()

        sims = np.zeros((len(docs), len(docs)))
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                min_sum = np.minimum(X[i], X[j]).sum()
                max_sum = np.maximum(X[i], X[j]).sum()
                sims[i, j] = min_sum / max_sum if max_sum > 0 else 0.0

        relationships = []
        for i in range(len(kg.nodes)):
            for j in range(i + 1, len(kg.nodes)):
                if sims[i, j] >= self.threshold:
                    relationships.append(
                        Relationship(
                            source=kg.nodes[i],
                            target=kg.nodes[j],
                            type="weighted_jaccard",
                            properties={self.new_property_name: float(sims[i, j])},
                        )
                    )
        return relationships
