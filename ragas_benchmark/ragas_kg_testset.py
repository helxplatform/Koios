#  Purpose: Generate test questions grounded in biomedical abstracts
from __future__ import annotations
import argparse, asyncio, json, inspect, random, re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
from dataclasses import dataclass
import typing as t

# --- RAGAS imports: core graph + transforms ---
from ragas.testset.graph import Node, Relationship, KnowledgeGraph, NodeType
from ragas.testset.transforms import apply_transforms, Parallel
from ragas.testset.transforms.relationship_builders.traditional import JaccardSimilarityBuilder
from ragas.testset.transforms.extractors import NERExtractor
from ragas.testset.synthesizers.multi_hop.base import MultiHopQuerySynthesizer, MultiHopScenario
from ragas.testset.synthesizers.prompts import ThemesPersonasInput, ThemesPersonasMatchingPrompt

import uuid

def safe_json(obj):
    """Safely convert non-serializable types like UUIDs."""
    if isinstance(obj, uuid.UUID):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


# Custom Multi-Hop Synthesizer (https://docs.ragas.io/en/latest/howtos/customizations/testgenerator/_testgen-customisation/#set-up-the-llm-and-embedding-model)
@dataclass
class MyMultiHopQuery(MultiHopQuerySynthesizer):
    theme_persona_matching_prompt = ThemesPersonasMatchingPrompt()

    async def _generate_scenarios(
        self, n: int, knowledge_graph: KnowledgeGraph, persona_list, callbacks
    ) -> t.List[MultiHopScenario]:
        print(f"[DEBUG] Running MyMultiHopQuery with {len(knowledge_graph.relationships)} total relationships...")

        results = [
            r for r in knowledge_graph.relationships
            if any(k in r.properties for k in ("entity_jaccard_similarity", "keyphrase_jaccard_similarity"))
        ]

        if not results:
            print("[WARN] No Jaccard-based edges found — using fallback relationships.")
            results = random.sample(knowledge_graph.relationships, min(50, len(knowledge_graph.relationships)))

        print(f"[DEBUG] Candidate multi-hop edges: {len(results)}")
        num_sample_per_rel = max(1, n // len(results))
        scenarios: list[MultiHopScenario] = []

        # Randomly sample to keep speed reasonable
        sampled_rels = random.sample(results, min(50, len(results)))

        for rel in sampled_rels:
            if len(scenarios) >= n:
                break

            node_a, node_b = rel.source, rel.target
            overlap_keys = []
            if "entity_jaccard_similarity" in rel.properties:
                overlap_keys.append("entities")
            if "keyphrase_jaccard_similarity" in rel.properties:
                overlap_keys.append("keyphrases")
            if not overlap_keys:
                overlap_keys.append("entities")

            # Log overlap info
            print(f"[DEBUG] Nodes connected: A({len(node_a.properties.get('entities', []))} ents,"
                  f" {len(node_a.properties.get('keyphrases', []))} keys) "
                  f"B({len(node_b.properties.get('entities', []))} ents,"
                  f" {len(node_b.properties.get('keyphrases', []))} keys) "
                  f"Overlap={overlap_keys}")

            # Persona cache
            cache_key = tuple(sorted(overlap_keys))
            if not hasattr(self, "_persona_cache"):
                self._persona_cache = {}
            if cache_key not in self._persona_cache:
                prompt_input = ThemesPersonasInput(themes=overlap_keys, personas=persona_list)
                persona_concepts = await self.theme_persona_matching_prompt.generate(
                    data=prompt_input, llm=self.llm, callbacks=callbacks
                )
                self._persona_cache[cache_key] = persona_concepts
            else:
                persona_concepts = self._persona_cache[cache_key]

            # Try to build multi-hop combinations
            base_scenarios = self.prepare_combinations(
                [node_a, node_b],
                overlap_keys,
                personas=persona_list,
                persona_item_mapping=persona_concepts.mapping,
                property_name=overlap_keys[0],
            )

            # Log results
            if not base_scenarios:
                print(f"[WARN] No combinations for {node_a.properties.get('title', '')[:50]} "
                      f"↔ {node_b.properties.get('title', '')[:50]}")
                s = MultiHopScenario(
                    nodes=[node_a, node_b],
                    reasoning="Explores a relationship between two related biomedical abstracts.",
                    synthesizer_name="multi_hop_query",
                    style="analytical",
                    length="medium",
                    persona=random.choice(persona_list),     
                    combinations=[
                        {
                            "property_name": "entities",
                            "overlap_type": "jaccard_similarity",
                            "value": rel.properties.get("entity_jaccard_similarity", 0.0)
                        }
                    ],
                )
                base_scenarios = [s]

            base_scenarios = self.sample_diverse_combinations(base_scenarios, num_sample_per_rel)
            for s in base_scenarios:
                s.synthesizer_name = "multi_hop_query"
            scenarios.extend(base_scenarios)

        print(f"[INFO] Generated {len(scenarios)} multi-hop scenarios (optimized).")
        return scenarios

#  Data Loading and Preprocessing
def load_abstracts(path: str | Path) -> List[Dict[str, Any]]:
    """
    Load and normalize dataset of abstracts.
    Supports CSV, Excel, and JSON formats with dbGaP-like schema.
    """
    path = str(path).strip()
    if path.endswith(".csv"):
        print(f"[LOAD] Detected CSV → {path}")
        df = pd.read_csv(path, on_bad_lines="skip", engine="python")
        data = df.to_dict(orient="records")
    elif path.endswith(".xlsx"):
        print(f"[LOAD] Detected Excel → {path}")
        df = pd.read_excel(path)
        data = df.to_dict(orient="records")
    else:
        print(f"[LOAD] Detected JSON → {path}")
        data = json.loads(Path(path).read_text())

    items = [
        {
            "doc_id": row.get("Accession") or row.get("StudyId") or row.get("id"),
            "title": row.get("Study Name") or row.get("StudyName") or row.get("title") or "",
            "abstract": row.get("Description") or row.get("abstract") or "",
            "permalink": row.get("Permalink") or "",
        }
        for row in data
    ]
    print(f"[LOAD] Parsed {len(items)} documents.")
    return items


# =========================================================
#  Node Construction (Sentence Splitting)
# =========================================================
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def chunk_document_to_nodes(doc: Dict[str, Any]) -> List[Node]:
    """Split a document into sentence-level nodes and merge short fragments."""
    title = (doc.get("title") or "").strip()
    abstract = (doc.get("abstract") or "").strip()
    text = (title + ". " + abstract).strip().strip(". ")
    if not text:
        return []

    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    merged, buffer = [], ""
    for s in sents:
        if len(s.split()) < 6:
            buffer += " " + s
            continue
        if buffer:
            s = buffer.strip() + " " + s
            buffer = ""
        merged.append(s)
    if buffer:
        merged.append(buffer.strip())

    return [
        Node(properties={
            "page_content": sent,
            "doc_id": doc.get("doc_id"),
            "title": title,
            "permalink": doc.get("permalink", ""),
            "position": i,
        })
        for i, sent in enumerate(merged)
    ]

def build_nodes(rows: List[Dict[str, Any]]) -> List[Node]:
    """Flatten all document nodes into a single list."""
    return [n for r in rows for n in chunk_document_to_nodes(r)]


#  Graph Enrichment (NER, Keyphrases, and Jaccard)
async def enrich_graph_with_transforms(nodes: List[Node], outdir: Path) -> Tuple[KnowledgeGraph, List]:
    """Apply entity/keyphrase extraction, clean entities, and compute Jaccard similarities."""
    outdir.mkdir(parents=True, exist_ok=True)
    kg = KnowledgeGraph(nodes=nodes)

    # Add metadata
    for node in kg.nodes:
        node.properties["doc_source"] = node.properties.get("doc_id", "unknown")

    # Remove old entity/keyphrase fields if any
    for n in kg.nodes:
        for k in ("entities", "keyphrases"):
            n.properties.pop(k, None)

    # --- Safe JSON encoder for UUIDs ---
    def safe_json(obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    # --- Save initial graph snapshot (before transforms) ---
    graph_data = {
        "nodes": [
            {
                "id": str(getattr(node, "id", None)),
                "type": str(getattr(node, "type", None)),
                "properties": node.properties,
            }
            for node in kg.nodes
        ],
        "relationships": []
    }
    (outdir / "knowledge_graph_initial.json").write_text(
        json.dumps(graph_data, indent=2, default=safe_json)
    )

    # --- Run NER + keyphrase extraction ---
    from ragas.testset.transforms.extractors import KeyphrasesExtractor
    ner = NERExtractor()
    kx = KeyphrasesExtractor()
    extractor_block = Parallel(ner, kx)
    maybe_coro = apply_transforms(kg, [extractor_block])
    if inspect.isawaitable(maybe_coro):
        await maybe_coro

    # --- Clean entities ---
    STOP_ENTS = {"the", "this", "that", "for", "of", "in", "to", "and", "on", "by", "from", "with"}
    def clean_entity(e: str) -> str:
        e = e.strip().strip(".").strip(":")
        if not e or e.lower() in STOP_ENTS:
            return ""
        return e

    for node in kg.nodes:
        ents = node.properties.get("entities", [])
        cleaned = [clean_entity(e) for e in ents if clean_entity(e)]
        node.properties["entities"] = cleaned
        node.type = NodeType.CHUNK if cleaned else NodeType.DOCUMENT

    # --- Compute Jaccard relationships ---
    jaccard_transforms = [
        JaccardSimilarityBuilder(property_name="entities", new_property_name="entity_jaccard_similarity"),
        JaccardSimilarityBuilder(property_name="keyphrases", new_property_name="keyphrase_jaccard_similarity"),
    ]
    maybe_coro = apply_transforms(kg, jaccard_transforms)
    rels = await maybe_coro if inspect.isawaitable(maybe_coro) else maybe_coro

    # --- Cross-document tags ---
    cross_edges = []
    for r in kg.relationships:
        src_doc = r.source.properties.get("doc_source")
        tgt_doc = r.target.properties.get("doc_source")
        r.properties["cross_doc"] = src_doc != tgt_doc
        if src_doc != tgt_doc:
            cross_edges.append(r)

    #  Save final enriched knowledge graph 
    graph_data_final = {
        "nodes": [
            {
                "id": str(getattr(node, "id", None)),
                "type": str(getattr(node, "type", None)),
                "properties": node.properties,
            }
            for node in kg.nodes
        ],
        "relationships": [
            {
                "source": str(getattr(rel.source, "id", None)),
                "target": str(getattr(rel.target, "id", None)),
                "properties": rel.properties,
            }
            for rel in kg.relationships
        ],
    }
    (outdir / "knowledge_graph.json").write_text(
        json.dumps(graph_data_final, indent=2, default=safe_json)
    )

    print(f"[INFO] Cross-abstract edges: {len(cross_edges)} / {len(kg.relationships)} total")
    print(f"[DEBUG] Sample relationship keys: {[list(r.properties.keys()) for r in kg.relationships[:3]]}")
    return kg, rels or kg.relationships

#  Main: Testset Generation
async def _amain(args):
    rows = load_abstracts(args.input)
    nodes = build_nodes(rows)
    kg, _ = await enrich_graph_with_transforms(nodes, Path(args.outdir))

    from ragas.testset import TestsetGenerator
    from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import OpenAIEmbeddings
    from langchain_openai import ChatOpenAI
    import openai
    from ragas.testset.persona import Persona

    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    openai_client = openai.OpenAI()
    generator_embeddings = OpenAIEmbeddings(client=openai_client, model="text-embedding-3-small")

    personas = [
        Persona(name="Clinician", role_description="Medical professional interpreting research findings."),
        Persona(name="Data Scientist", role_description="Interested in modeling and computational aspects."),
        Persona(name="Graduate Student", role_description="Learning research methods."),
    ]

    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="entities"), 0.35),
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="keyphrases"), 0.35),
        (MyMultiHopQuery(llm=generator_llm), 0.30),
    ]

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=kg,
        persona_list=personas,
    )

    print("[INFO] Generating testset (with fallback for multi-hop)...")
    testset = generator.generate(testset_size=args.n_samples, query_distribution=query_distribution)
    df = testset.to_pandas()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_json(outdir / "ragas_testset.jsonl", orient="records", lines=True)

    print(f"[KG] nodes={len(kg.nodes)} relationships={len(kg.relationships)}")
    print(f"[TESTSET] samples={len(df)} → {outdir/'ragas_testset.jsonl'}")


#  CLI Entry Point
def main():
    p = argparse.ArgumentParser(description="Generate RAGAS-style testset from abstracts.")
    p.add_argument("--input", required=True, help="Path to JSON/CSV/Excel of abstracts.")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--n-samples", type=int, default=40, help="Number of samples to generate.")
    args = p.parse_args()
    asyncio.run(_amain(args))

if __name__ == "__main__":
    main()
