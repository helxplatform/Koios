from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np
import math
import typing as t
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from ragas.testset.transforms.base import RelationshipBuilder, KnowledgeGraph, Relationship


@dataclass
class HybridRelationshipBuilder(RelationshipBuilder):
    property_name: str = "entities"
    new_property_name: str = "hybrid_similarity"
    n_clusters: int = 8
    jaccard_weight: float = 0.6
    pmi_weight: float = 0.4
    threshold: float = 0.25

    async def transform(self, kg: KnowledgeGraph) -> t.List[Relationship]:
        # 1️⃣  Prepare TF-IDF representation of each node’s entity list
        docs = [" ".join(n.get_property(self.property_name) or []) for n in kg.nodes]
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w-]+\b")
        X = vectorizer.fit_transform(docs).toarray()
        vocab = vectorizer.get_feature_names_out()

        # 2️⃣  Compute pairwise Weighted Jaccard similarities
        n = len(kg.nodes)
        weighted_jaccard = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                min_sum = np.minimum(X[i], X[j]).sum()
                max_sum = np.maximum(X[i], X[j]).sum()
                sim = min_sum / max_sum if max_sum > 0 else 0.0
                weighted_jaccard[i, j] = weighted_jaccard[j, i] = sim

        # 3️⃣  Build PMI matrix between *terms*, then lift to node-level
        all_terms = [t for n_ in kg.nodes for t in (n_.get_property(self.property_name) or [])]
        term_counts = Counter(all_terms)
        total_terms = sum(term_counts.values())

        pair_counts = Counter()
        for n_ in kg.nodes:
            ents = list(set(n_.get_property(self.property_name) or []))
            for a in range(len(ents)):
                for b in range(a + 1, len(ents)):
                    pair_counts[tuple(sorted([ents[a], ents[b]]))] += 1

        term_pmi = {}
        for (a, b), c_ab in pair_counts.items():
            p_a = term_counts[a] / total_terms
            p_b = term_counts[b] / total_terms
            p_ab = c_ab / len(kg.nodes)
            if p_ab > 0 and p_a > 0 and p_b > 0:
                pmi = math.log2(p_ab / (p_a * p_b))
                term_pmi[(a, b)] = max(pmi, 0)

        # node-level PMI similarity (average pair PMI of overlapping entities)
        pmi_sim = np.zeros((n, n))
        for i in range(n):
            ents_i = set(kg.nodes[i].get_property(self.property_name) or [])
            for j in range(i + 1, n):
                ents_j = set(kg.nodes[j].get_property(self.property_name) or [])
                overlap = ents_i & ents_j
                if not overlap:
                    continue
                vals = []
                for a in overlap:
                    for b in overlap:
                        if a == b:
                            continue
                        key = tuple(sorted([a, b]))
                        if key in term_pmi:
                            vals.append(term_pmi[key])
                if vals:
                    sim = np.mean(vals)
                    pmi_sim[i, j] = pmi_sim[j, i] = sim

        # 4️⃣  Cluster nodes for localized edges
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        centroid_sim = cosine_similarity(centroids)

        # 5️⃣  Combine scores and threshold
        relationships = []
        for i in range(n):
            for j in range(i + 1, n):
                # mix Weighted Jaccard + PMI
                combined = (
                    self.jaccard_weight * weighted_jaccard[i, j]
                    + self.pmi_weight * pmi_sim[i, j]
                )

                # mild cluster bonus: same cluster or close centroids
                if labels[i] == labels[j]:
                    combined += 0.1
                else:
                    combined += 0.05 * centroid_sim[labels[i], labels[j]]

                if combined >= self.threshold:
                    relationships.append(
                        Relationship(
                            source=kg.nodes[i],
                            target=kg.nodes[j],
                            type="hybrid_similarity",
                            properties={self.new_property_name: float(combined)},
                        )
                    )
        return relationships
