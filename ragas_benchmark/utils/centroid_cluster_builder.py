from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from ragas.testset.transforms.base import RelationshipBuilder, KnowledgeGraph, Relationship
from dataclasses import dataclass

@dataclass
class ClusterCentroidBuilder(RelationshipBuilder):
    property_name: str = "entities"
    new_property_name: str = "cluster_similarity"
    n_clusters: int = 14
    intra_threshold: float = 0.8

    async def transform(self, kg: KnowledgeGraph):
        docs = [" ".join(n.get_property(self.property_name) or []) for n in kg.nodes]
        vec = TfidfVectorizer().fit(docs)
        X = vec.transform(docs)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        sims = cosine_similarity(centroids)

        relationships = []
        for i, node in enumerate(kg.nodes):
            for j, node2 in enumerate(kg.nodes):
                if i >= j:
                    continue
                if labels[i] == labels[j]:
                    relationships.append(
                        Relationship(
                            source=node,
                            target=node2,
                            type="intra_cluster",
                            properties={"similarity": 1.0},
                        )
                    )
                else:
                    sim = sims[labels[i], labels[j]]
                    if sim >= self.intra_threshold:
                        relationships.append(
                            Relationship(
                                source=node,
                                target=node2,
                                type="inter_cluster",
                                properties={"similarity": float(sim)},
                            )
                        )
        return relationships
