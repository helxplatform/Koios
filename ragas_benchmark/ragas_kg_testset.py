"""
Purpose:
--------
Generate test questions (scenarios) grounded in biomedical abstracts.
This script:
  1. Loads a dataset of biomedical abstracts (CSV, JSON, or Excel).
  2. Extracts entities (NER) using SciSpacy.
  3. Builds a knowledge graph with relationships (Jaccard similarity, etc.).
  4. Automatically detects relation types and creates multi-hop question scenarios.
  5. Uses an LLM (GPT-4o-mini) and embeddings to generate a RAGAS-style testset.
  6. Outputs the full testset + per-relation-type subsets as JSONL files.
"""

from __future__ import annotations
import argparse, asyncio, json, inspect, random, re, uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
from dataclasses import dataclass
import typing as t

# --- RAGAS imports (core framework for testset generation) ---
from ragas.testset.graph import Node, KnowledgeGraph, NodeType
from ragas.testset.transforms import apply_transforms, Parallel
from ragas.testset.transforms.relationship_builders.traditional import JaccardSimilarityBuilder
from ragas.testset.synthesizers.multi_hop.base import MultiHopQuerySynthesizer, MultiHopScenario
from ragas.testset.synthesizers.prompts import ThemesPersonasInput, ThemesPersonasMatchingPrompt

# ----------------------------------------------------------------------
# Utility: JSON serialization helper for UUIDs
# ----------------------------------------------------------------------
def safe_json(obj):
    """Converts UUID objects to strings when dumping JSON, 
    avoiding serialization errors."""
    if isinstance(obj, uuid.UUID):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


# ----------------------------------------------------------------------
# Custom Multi-Hop Query Synthesizer (auto-detect relation types)
# ----------------------------------------------------------------------

@dataclass
class MyMultiHopQuery(MultiHopQuerySynthesizer):
    """
    Extends RAGAS's MultiHopQuerySynthesizer to automatically
    generate multi-hop scenarios from biomedical abstracts.

    Attributes:
    -----------
    documents : list
        List of Node objects containing biomedical abstracts.
    """

    def __init__(self, documents=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.documents = documents or []

    async def _generate_scenarios(
        self,
        docs=None,
        sample_generation_grp=None,
        persona=None,
        synthesizer_name=None,
        n_samples=None,
    ):
        """
        Generates multi-hop scenarios between document pairs.

        Steps:
        1. Checks and loads documents.
        2. Randomly selects document pairs.
        3. Builds MultiHopScenario objects (question-like examples).
        """

        # --- Validate docs input ---
        if not isinstance(docs, (list, tuple)):
            print(f"[WARN] Expected docs list, got {type(docs).__name__} = {docs}")
            docs = getattr(self, "documents", [])
            print(f"[INFO] Falling back to self.documents ({len(docs)} items)")

        print(f"[INFO] Generating multi-hop scenarios from {len(docs)} documents...")

        # --- Sample document pairs ---
        n_docs = len(docs)
        n_pairs = min(n_samples or 10, n_docs * (n_docs - 1) // 2)
        pairs = set()

        # Randomly choose unique document pairs
        while len(pairs) < n_pairs:
            i, j = random.sample(range(n_docs), 2)
            if i != j:
                pairs.add(tuple(sorted((i, j))))
        pairs = list(pairs)

        # --- Create multi-hop scenarios ---
        all_scenarios = []
        for (i, j) in pairs:
            d1, d2 = docs[i], docs[j]
            sim = 0.5  # Placeholder similarity score

            # Create one scenario describing the relationship between d1 and d2
            scenario = MultiHopScenario(
                nodes=[d1, d2],
                source_documents=[d1, d2],
                relation_type="entity_jaccard_similarity",
                score=sim,
                combinations=[f"{d1.properties.get('title', 'Doc A')} ↔ {d2.properties.get('title', 'Doc B')}"],
                style="Perfect grammar",
                length="long",
                persona={
                    "name": "Biomedical Researcher",
                    "role_description": "Explores relationships between studies and genetic traits"
                },
            )
            all_scenarios.append(scenario)

        print(f"[INFO] Created {len(all_scenarios)} sampled MultiHopScenario objects.")
        return all_scenarios


# ----------------------------------------------------------------------
# Helpers: Loading and preprocessing biomedical abstracts
# ----------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def load_abstracts(path: str | Path) -> List[Dict[str, Any]]:
    """
    Loads biomedical abstracts from a CSV, JSON, or Excel file.
    Returns a list of dictionaries with fields:
      - doc_id
      - title
      - abstract
      - permalink
    """
    path = str(path).strip()
    if path.endswith(".csv"):
        df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    elif path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.DataFrame(json.loads(Path(path).read_text()))

    print(f"[LOAD] Parsed {len(df)} documents.")

    # Normalize columns from different dataset formats
    return [
        {
            "doc_id": row.get("Accession") or row.get("StudyId") or row.get("id"),
            "title": row.get("Study Name") or row.get("StudyName") or row.get("title") or "",
            "abstract": row.get("Description") or row.get("abstract") or "",
            "permalink": row.get("Permalink") or "",
        }
        for _, row in df.iterrows()
    ]


def chunk_document_to_nodes(doc: Dict[str, Any]) -> List[Node]:
    """
    Converts a document (title + abstract) into a single Node object.
    If the document text is empty, returns an empty list.
    """
    title = (doc.get("title") or "").strip()
    abstract = (doc.get("abstract") or "").strip()
    text = (title + ". " + abstract).strip().strip(". ")

    if not text:
        return []

    return [
        Node(
            properties={
                "page_content": text,
                "doc_id": doc.get("doc_id"),
                "title": title,
                "permalink": doc.get("permalink", ""),
            }
        )
    ]


def build_nodes(rows: List[Dict[str, Any]]) -> List[Node]:
    """Convert a list of document rows into a flat list of Node objects."""
    return [n for r in rows for n in chunk_document_to_nodes(r)]


# ----------------------------------------------------------------------
# Graph enrichment: Named Entity Recognition (NER) + relationships
# ----------------------------------------------------------------------

async def enrich_graph_with_transforms(nodes: List[Node], outdir: Path):
    """
    Adds semantic enrichment to the knowledge graph:
    - Extracts biomedical entities using SciSpacy.
    - Builds entity-based similarity relationships (Jaccard).
    - Marks cross-document edges.
    """

    outdir.mkdir(parents=True, exist_ok=True)
    kg = KnowledgeGraph(nodes=nodes)

    # --- Named Entity Recognition ---
    from utils.scispacyNER import SciSpacyNERExtractor
    ner = SciSpacyNERExtractor()
    extractor_block = Parallel(ner)

    maybe_coro = apply_transforms(kg, [extractor_block])
    if inspect.isawaitable(maybe_coro):
        await maybe_coro

    # --- Clean extracted entities ---
    STOP_ENTS = {"the","this","that","for","of","in","to","and","on","by","from","with"}
    def clean_entity(e: str):
        e = e.strip().strip(".").strip(":")
        return e if e and e.lower() not in STOP_ENTS else ""

    for node in kg.nodes:
        ents = node.properties.get("entities", [])
        node.properties["entities"] = [clean_entity(e) for e in ents if clean_entity(e)]
        node.type = NodeType.CHUNK if node.properties["entities"] else NodeType.DOCUMENT

    # --- Build similarity relationships ---
    jaccard_transforms = [
        JaccardSimilarityBuilder(property_name="entities", new_property_name="entity_jaccard_similarity"),
        JaccardSimilarityBuilder(property_name="keyphrases", new_property_name="keyphrase_jaccard_similarity"),
    ]
    maybe_coro = apply_transforms(kg, jaccard_transforms)
    if inspect.isawaitable(maybe_coro):
        await maybe_coro

    # --- Label cross-document edges ---
    for r in kg.relationships:
        src_doc, tgt_doc = r.source.properties.get("doc_id"), r.target.properties.get("doc_id")
        r.properties["cross_doc"] = src_doc != tgt_doc

    print(f"[INFO] Cross-abstract edges: {sum(r.properties['cross_doc'] for r in kg.relationships)} / {len(kg.relationships)} total")
    return kg


# ----------------------------------------------------------------------
# Main Async Routine
# ----------------------------------------------------------------------

async def _amain(args):
    """Core async pipeline to build KG and generate the testset."""

    # --- Load and preprocess data ---
    rows = load_abstracts(args.input)
    nodes = build_nodes(rows)

    # --- Enrich KG with NER + relationships ---
    kg = await enrich_graph_with_transforms(nodes, Path(args.outdir))

    print(f"Total nodes: {len(kg.nodes)}")
    print(f"Total relationships: {len(kg.relationships)}")

    # Preview first few relationships
    for i, r in enumerate(kg.relationships[:5]):
        print(f"Edge {i}: {r.source.properties.get('title')} ↔ {r.target.properties.get('title')}")
        print(f"  Properties: {r.properties}")

    # --- Initialize LLM and embeddings for testset generation ---
    from ragas.testset import TestsetGenerator
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import OpenAIEmbeddings
    from langchain_openai import ChatOpenAI
    import openai
    from ragas.testset.persona import Persona

    # Wrap GPT-4o-mini for LLM reasoning
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    openai_client = openai.OpenAI()
    generator_embeddings = OpenAIEmbeddings(client=openai_client, model="text-embedding-3-small")

    # Define personas (roles) for generating diverse questions
    personas = [
        Persona(name="Clinician", role_description="Medical professional interpreting research findings."),
        Persona(name="Data Scientist", role_description="Interested in modeling and computational aspects."),
        Persona(name="Graduate Student", role_description="Learning research methods."),
    ]

    # Use our custom multi-hop synthesizer for generating queries
    query_distribution = [(MyMultiHopQuery(documents=nodes, llm=generator_llm), 1.0)]

    # --- Generate the RAGAS-style testset ---
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=kg,
        persona_list=personas,
    )

    print("[INFO] Generating testset (multi-hop across relation types)...")
    testset = generator.generate(testset_size=args.n_samples, query_distribution=query_distribution)
    df = testset.to_pandas()

    # --- Save testset to output directory ---
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_json(outdir / "ragas_testset.jsonl", orient="records", lines=True)
    print(f"[TESTSET] total samples={len(df)} → {outdir/'ragas_testset.jsonl'}")

    # --- Optionally save per-relation-type subsets ---
    synth = query_distribution[0][0]
    if hasattr(synth, "_scenarios_by_type"):
        for rel_type, scenarios in synth._scenarios_by_type.items():
            subpath = outdir / f"ragas_testset_{rel_type}.jsonl"
            subset = []
            for s in scenarios:
                record = s.model_dump()
                record["nodes"] = [
                    {
                        "doc_id": n.properties.get("doc_id"),
                        "title": n.properties.get("title"),
                        "permalink": n.properties.get("permalink"),
                        "entities": n.properties.get("entities", [])[:10],
                    }
                    for n in getattr(s, "nodes", [])
                ]
                subset.append(record)

            with open(subpath, "w") as f:
                for row in subset:
                    f.write(json.dumps(row, default=safe_json) + "\n")

            print(f"[SAVE] {rel_type}: {len(subset)} samples → {subpath}")


# ----------------------------------------------------------------------
# Entry point for CLI
# ----------------------------------------------------------------------

def main():
    """Command-line interface for running the full pipeline."""
    p = argparse.ArgumentParser(description="Generate RAGAS-style testset from biomedical abstracts.")
    p.add_argument("--input", required=True, help="Path to input CSV/JSON/Excel file.")
    p.add_argument("--outdir", required=True, help="Output directory for testset files.")
    p.add_argument("--n-samples", type=int, default=40, help="Number of samples to generate.")
    args = p.parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
