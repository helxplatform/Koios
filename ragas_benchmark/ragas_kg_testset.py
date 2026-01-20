"""
Modern RAGAS KG Testset Generator (2025 API)
------------------------------------------------
✓ SciSpacy NER
✓ Weighted Jaccard
✓ Jaccard Similarity
✓ Centroid Similarity
✓ Single-Hop + Multi-Hop QA
✓ Fully compatible with RAGAS 2025 API
"""

from __future__ import annotations
import argparse, asyncio, json, uuid, random
from pathlib import Path
from typing import Any, Dict, List
from dataclasses import dataclass
import pandas as pd

# --------------------------------------------------------
# RAGAS modern imports
# --------------------------------------------------------
from ragas.testset.graph import Node, KnowledgeGraph, NodeType
from ragas.testset.transforms import apply_transforms, Parallel
from ragas.testset.synthesizers.single_hop import (
    SingleHopQuerySynthesizer,
    SingleHopScenario,
)
from ragas.testset.synthesizers.multi_hop import (
    MultiHopQuerySynthesizer,
    MultiHopScenario,
)
from ragas.testset import TestsetGenerator
from ragas.llms.base import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.testset.persona import Persona

# --------------------------------------------------------
# Your custom builders
# --------------------------------------------------------
from ragas_benchmark.utils.scispacyNER import SciSpacyNERExtractor
from ragas_benchmark.utils.jaccardTFIDF import WeightedJaccardBuilder
from ragas_benchmark.utils.centroid_cluster_builder import ClusterCentroidBuilder
from ragas.testset.transforms.relationship_builders.traditional import (
    JaccardSimilarityBuilder,
)

# --------------------------------------------------------
# Helpers
# --------------------------------------------------------
def safe_json(obj):
    if isinstance(obj, uuid.UUID):
        return str(obj)
    raise TypeError(f"Cannot serialize {type(obj)}")


QUESTION_TEMPLATES = {
    "entity_jaccard_similarity": (
        "What shared biomedical entities link '{a}' and '{b}'?",
        "These abstracts share overlapping biomedical entities.",
    ),
    "weighted_jaccard_similarity": (
        "How are '{a}' and '{b}' related based on high-importance overlapping entities?",
        "Weighted Jaccard similarity indicates strong shared semantic features.",
    ),
    "centroid_similarity": (
        "How do '{a}' and '{b}' align conceptually?",
        "Their embeddings place them in similar conceptual clusters.",
    ),
    "default": (
        "What is the relationship between '{a}' and '{b}'?",
        "They show semantic overlap.",
    ),
}

# --------------------------------------------------------
# SINGLE-HOP GENERATOR
# --------------------------------------------------------
@dataclass
class MySingleHopQuery(SingleHopQuerySynthesizer):
    kg: KnowledgeGraph = None

    async def _generate_scenarios(
        self,
        docs=None,
        sample_generation_grp=None,
        persona=None,
        synthesizer_name=None,
        n_samples=None,
    ):

        # --- FIX persona (RAGAS sometimes passes [Persona]) ---
        if isinstance(persona, list) and len(persona) > 0:
            persona = persona[0]

        n_samples = n_samples or 10
        n_samples = min(n_samples, len(self.kg.relationships))

        edges = random.sample(self.kg.relationships, n_samples)
        scenarios = []

        for r in edges:
            src, tgt = r.source, r.target
            ta = src.properties.get("title", "Document A")
            tb = tgt.properties.get("title", "Document B")

            # choose similarity metric
            rel_key = next(
                (k for k in r.properties if "similarity" in k), "default"
            )
            q_tpl, a_tpl = QUESTION_TEMPLATES.get(
                rel_key, QUESTION_TEMPLATES["default"]
            )

            scenarios.append(
                SingleHopScenario(
                    term=f"{ta} ↔ {tb}",
                    question=q_tpl.format(a=ta, b=tb),
                    answer=a_tpl,
                    nodes=[src, tgt],
                    context_documents=[src, tgt],
                    style="Perfect grammar",
                    length="short",
                    persona=persona,
                )
            )

        return scenarios


# --------------------------------------------------------
# MULTI-HOP GENERATOR
# --------------------------------------------------------
@dataclass
class MyMultiHopQuery(MultiHopQuerySynthesizer):
    documents: List[Node] = None

    async def _generate_scenarios(
        self,
        docs=None,
        sample_generation_grp=None,
        persona=None,
        synthesizer_name=None,
        n_samples=None,
    ):

        # fix persona = [Persona] case
        if isinstance(persona, list) and len(persona) > 0:
            persona = persona[0]

        # fix docs=int bug
        if not isinstance(docs, list):
            docs = self.documents

        n_docs = len(docs)
        n_samples = n_samples or 10

        scenarios = []

        for _ in range(n_samples):
            i, j = random.sample(range(n_docs), 2)
            d1, d2 = docs[i], docs[j]

            t1 = d1.properties.get("title", "Doc A")
            t2 = d2.properties.get("title", "Doc B")
            id1 = str(d1.properties.get("doc_id") or d1.properties.get("title") or "A")
            id2 = str(d2.properties.get("doc_id") or d2.properties.get("title") or "B")

            scenario = MultiHopScenario(
                term=f"{t1} ↔ {t2}",
                question=f"How do findings from '{t1}' connect to '{t2}' via multi-step biomedical reasoning?",
                answer="The studies provide complementary insights connected through multi-step reasoning.",
                nodes=[d1, d2],
                context_documents=[d1, d2],
                path=[d1, d2],

                combinations=[f"{id1} | {id2}"],


                style="Perfect grammar",
                length="long",
                persona=persona,
                extra_info={
                        "doc_ids": [
                            d1.properties.get("doc_id"),
                            d2.properties.get("doc_id")
                        ],
                        "titles": [t1, t2],
                        "entities_A": ents1,
                        "entities_B": ents2,
                        "edge_type": key,
                        "edge_properties": props,
                }
            )



            scenarios.append(scenario)

        return scenarios


# --------------------------------------------------------
# LOADING + NODE BUILDING
# --------------------------------------------------------
def load_abstracts(path: str | Path) -> List[Dict[str, Any]]:
    path = str(path)
    df = (
        pd.read_csv(path, on_bad_lines="skip", engine="python")
        if path.endswith(".csv")
        else pd.read_excel(path)
    )

    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "doc_id": row.get("Accession") or row.get("StudyId"),
                "title": row.get("Study Name") or "",
                "abstract": row.get("Description") or "",
                "permalink": row.get("Permalink") or "",
            }
        )
    return rows


def chunk_document_to_nodes(doc: Dict[str, Any]) -> List[Node]:
    text = f"{doc['title']}. {doc['abstract']}".strip()
    if not text:
        return []

    return [
        Node(
            properties={
                "page_content": text,
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "permalink": doc.get("permalink", ""),
            }
        )
    ]


def build_nodes(rows: List[Dict[str, Any]]) -> List[Node]:
    return [n for r in rows for n in chunk_document_to_nodes(r)]


# --------------------------------------------------------
# KG TRANSFORMS
# --------------------------------------------------------
async def enrich_graph_with_transforms(nodes: List[Node], outdir: Path):
    kg = KnowledgeGraph(nodes=nodes)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- NER ---
    ner = SciSpacyNERExtractor()
    maybe = apply_transforms(kg, [Parallel(ner)])
    if asyncio.iscoroutine(maybe):
        await maybe

    # --- Clean entities ---
    STOP = {"the", "a", "an", "on", "in", "of", "to", "and"}
    for n in kg.nodes:
        ents = [
            e.strip().lower()
            for e in n.properties.get("entities", [])
            if e.lower() not in STOP
        ]
        n.properties["entities"] = ents
        n.type = NodeType.CHUNK if ents else NodeType.DOCUMENT

    # --- Similarity & centroid transforms ---
    transforms = [
        WeightedJaccardBuilder("entities", "weighted_jaccard_similarity", 0.5),
        JaccardSimilarityBuilder("entities", "entity_jaccard_similarity", 0.5),
        ClusterCentroidBuilder("entities", "centroid_similarity"),
    ]

    maybe = apply_transforms(kg, transforms)
    if asyncio.iscoroutine(maybe):
        await maybe

    return kg


# --------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------
async def _amain(args):

    # --- Load + Build Nodes ---
    rows = load_abstracts(args.input)
    nodes = build_nodes(rows)

    # --- Build KG ---
    kg = await enrich_graph_with_transforms(nodes, Path(args.outdir))

    # --- LLM + Embeddings ---
    llm = llm_factory("gpt-4o-mini")

    from openai import OpenAI
    client = OpenAI()
    embeddings = OpenAIEmbeddings(client=client, model="text-embedding-3-small")

    personas = [
        Persona(name="Clinician", role_description="Interprets biomedical findings."),
        Persona(name="Data Scientist", role_description="Understands modeling."),
        Persona(name="Graduate Student", role_description="Learning research."),
    ]

    # --- Query Distribution ---
    query_distribution = [
        (MySingleHopQuery(kg=kg, llm=llm), 0.4),
        (MyMultiHopQuery(documents=nodes, llm=llm), 0.6),
    ]

    # --- Generate Testset ---
    generator = TestsetGenerator(
        llm=llm,
        embedding_model=embeddings,
        knowledge_graph=kg,
        persona_list=personas,
    )

    print("[INFO] Generating RAGAS Testset...")
    testset = generator.generate(
        args.n_samples,
        query_distribution=query_distribution,
    )

    df = testset.to_pandas()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df.to_json(out / "ragas_testset.jsonl", orient="records", lines=True)
    print(f"[DONE] Wrote {len(df)} samples → {out/'ragas_testset.jsonl'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--n-samples", type=int, default=40)
    args = p.parse_args()

    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
