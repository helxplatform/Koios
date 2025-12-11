# MULTI-HOP TESTSET GENERATOR FOR BIO-MEDICAL ABSTRACTS (RAGAS)
#
# We:
#   1. Loads dbGaP-style biomedical study metadata from CSV
#   2. Converts documents to RAGAS Node objects
#   3. Builds a Knowledge Graph using:
#        - SciSpaCy NER
#        - Entity-based similarity relationships
#        - TF-IDF weighted Jaccard similarity
#        - Centroid-based embedding similarity
#   4. Uses a custom MultiHopQuerySynthesizer to generate multi-hop
#      biomedical reasoning questions grounded in KG edges.
#
# Output: JSONL dataset of multi-hop question/answer scenarios.

from __future__ import annotations
import argparse, asyncio, random
from pathlib import Path
from dataclasses import dataclass
import pandas as pd

# LangChain for prompts + messages
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# RAGAS graph primitives
from ragas.testset.graph import Node, KnowledgeGraph, NodeType

# RAGAS transform pipeline utilities
from ragas.testset.transforms import apply_transforms, Parallel

# RAGAS multi-hop generator classes
from ragas.testset.synthesizers.multi_hop import (
    MultiHopQuerySynthesizer,
    MultiHopScenario,
)

# RAGAS high-level testset wrapper
from ragas.testset import TestsetGenerator

# Embeddings + LLM factory wrappers
from ragas.llms.base import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.testset.persona import Persona

# Custom entity similarity + embedding similarity
from ragas_benchmark.utils.scispacyNER import SciSpacyNERExtractor
from ragas_benchmark.utils.jaccardTFIDF import WeightedJaccardBuilder
from ragas_benchmark.utils.centroid_cluster_builder import ClusterCentroidBuilder

# Classical similarity transform
from ragas.testset.transforms.relationship_builders.traditional import (
    JaccardSimilarityBuilder,
)

# OpenAI client for embeddings
from openai import OpenAI


# LOAD CSV → BASIC ROW DICTIONARIES
def load_rows(path):
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")  # safe parse

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "doc_id": r.get("Accession") or r.get("StudyId"),  # PHs identifier
            "title": r.get("Study Name", ""),
            "abstract": r.get("Description", ""),
            "permalink": r.get("Permalink", ""),
        })
    return rows


# BUILD NODE OBJECTS FOR RAGAS KNOWLEDGE GRAPH
def build_nodes(rows):
    nodes = []
    for r in rows:
        phs = r["doc_id"]
        title = r["title"]
        abstract = r["abstract"]

        # This is what the LLM *sees* as a document's full text
        page_text = (
            f"[PHS-ID: {phs}]\n"
            f"[TITLE: {title}]\n\n"
            f"{abstract}"
        )

        # Build the RAGAS Node with metadata
        node = Node(
            properties={
                "page_content": page_text,   # RAGAS uses this as context text
                "doc_id": phs,
                "title": title,
                "abstract": abstract,
                "permalink": r["permalink"],
            }
        )

        nodes.append(node)

    return nodes


# CUSTOM MULTI-HOP SYNTHESIS (OVERRIDES RAGAS DEFAULT)
@dataclass
class MyMultiHopQuery(MultiHopQuerySynthesizer):
    documents: list = None      # all KG nodes
    kg: KnowledgeGraph = None   # KG to traverse

    async def _generate_scenarios(
        self,
        docs=None,
        sample_generation_grp=None,
        persona=None,
        synthesizer_name=None,
        n_samples=None,
        **kwargs
    ):
        # RAGAS sometimes passes persona as list, normalize that
        if isinstance(persona, list):
            persona = persona[0]

        docs = docs if isinstance(docs, list) else self.documents
        n_samples = n_samples or 20

        scenarios = []

        # Build adjacency dictionary: node -> neighbors w/ edge properties
        adjacency = {}
        for rel in self.kg.relationships:
            a, b = rel.source, rel.target
            props = rel.properties

            # Create bidirectional mapping
            adjacency.setdefault(a, []).append((b, props))
            adjacency.setdefault(b, []).append((a, props))

        # LOOP: Generate N multi-hop QA examples
        for _ in range(n_samples):

            # Pick a random starting node
            d1 = random.choice(docs)
            neighbors = adjacency.get(d1, [])

            # Choose second node: actual linked or fallback
            if neighbors:
                d2, props = random.choice(neighbors)

                # Determine what kind of relationship is present
                if props.get("weighted_jaccard_similarity", 0) > 0:
                    key = "weighted_jaccard"
                elif props.get("entity_jaccard_similarity", 0) > 0:
                    key = "entity_jaccard"
                elif props.get("centroid_similarity", 0) > 0:
                    key = "centroid_embedding"
                else:
                    key = "linked"
            else:
                # No relationships → fallback random pairing
                d2 = random.choice([x for x in docs if x is not d1])
                key = "fallback_random"

            # Extract metadata for prompting
            titleA = d1.properties["title"]
            titleB = d2.properties["title"]
            phsA = d1.properties["doc_id"]
            phsB = d2.properties["doc_id"]
            absA = d1.properties["abstract"]
            absB = d2.properties["abstract"]
            entsA = d1.properties.get("entities", [])
            entsB = d2.properties.get("entities", [])

            # Randomly choose question length
            length = random.choice(["short", "long"])
            persona_role = persona.role_description if persona else "Biomedical expert"

            
            # LLM PROMPT TO GENERATE MULTI-HOP QA
            prompt = f"""
You are generating a MULTI-HOP biomedical QUESTION and ANSWER.

Persona: {persona.name} — {persona_role}
Desired question length: {length.upper()}
Multi-hop TYPE: {key}

STUDY A:
Title: {titleA}
Abstract: {absA}
Entities: {entsA}

STUDY B:
Title: {titleB}
Abstract: {absB}
Entities: {entsB}

TASK:
1. Produce a biomedical QUESTION requiring a multi-hop chain of reasoning across both abstracts.
2. Use TYPE to determine the reasoning link.
3. Then produce a CORRECT biomedical ANSWER grounded in both abstracts.

FORMAT:
QUESTION:
<your question>

ANSWER:
<your answer>
"""

            # Invoke LLM using LangChain-style wrappers
            template = ChatPromptTemplate.from_messages([HumanMessage(content=prompt)])
            chat_value = template.invoke({})
            raw = await self.llm.agenerate_text(chat_value)  # async call

            # Normalize output regardless of LLM wrapper format
            text = raw.generations[0][0].text if hasattr(raw, "generations") else str(raw)

            # Split into Q + A
            if "ANSWER:" in text:
                q, a = text.split("ANSWER:", 1)
                question = q.replace("QUESTION:", "").strip()
                answer = a.strip()
            else:
                question = text.strip()
                answer = "Model did not provide answer."

            # Store metadata back into nodes (kept by RAGAS)
            d1.properties["phs_id"] = phsA
            d1.properties["entities"] = entsA
            d1.properties["multi_hop_type"] = key
            d1.properties["persona"] = persona.name
            d1.properties["question_length"] = length

            d2.properties["phs_id"] = phsB
            d2.properties["entities"] = entsB
            d2.properties["multi_hop_type"] = key
            d2.properties["persona"] = persona.name
            d2.properties["question_length"] = length

            # Build MultiHopScenario (must match RAGAS API fields)
            scenario = MultiHopScenario(
                term=f"[{key}] {titleA} ↔ {titleB}",
                question=question,
                answer=answer,
                nodes=[d1, d2],                 # nodes involved in reasoning
                context_documents=[d1, d2],     # source documents
                path=[d1, d2],                  # hop path
                combinations=[f"{phsA} | {phsB}"],
                persona=persona,
                style="Perfect grammar",
                length=length,
            )

            scenarios.append(scenario)

        return scenarios


# BUILD THE KNOWLEDGE GRAPH (NER → SIMILARITIES)
async def enrich_graph(nodes, outdir: Path):
    kg = KnowledgeGraph(nodes=nodes)
    outdir.mkdir(parents=True, exist_ok=True)

    # STEP 1 — SciSpaCy NER applied in parallel
    ner = SciSpacyNERExtractor()
    maybe = apply_transforms(kg, [Parallel(ner)])
    if asyncio.iscoroutine(maybe):
        await maybe

    # Clean NER and assign node types
    STOP = {"the", "a", "an", "of", "in", "on", "to", "and"}
    for n in kg.nodes:
        ents = [e.lower().strip() for e in n.properties.get("entities", []) if e.lower() not in STOP]
        n.properties["entities"] = ents
        n.type = NodeType.CHUNK if ents else NodeType.DOCUMENT

    # STEP 2 — Add relationship builders (similarity edges)
    transforms = [
        WeightedJaccardBuilder("entities", "weighted_jaccard_similarity", 0.5),
        JaccardSimilarityBuilder("entities", "entity_jaccard_similarity", 0.5),
        ClusterCentroidBuilder("entities", "centroid_similarity"),
    ]

    maybe = apply_transforms(kg, transforms)
    if asyncio.iscoroutine(maybe):
        await maybe

    return kg


async def _amain(args):
    print("[INFO] Loading...")
    rows = load_rows(args.input)

    print("[INFO] Building nodes...")
    nodes = build_nodes(rows)

    print("[INFO] Running transforms...")
    kg = await enrich_graph(nodes, Path(args.outdir))

    print("[INFO] Init LLM + embeddings...")
    llm = llm_factory("gpt-4o-mini")
    embeddings = OpenAIEmbeddings(client=OpenAI(), model="text-embedding-3-small")

    # Personas influence prompt conditioning
    personas = [
        Persona(name="Clinician", role_description="Interprets biomedical results."),
        Persona(name="Researcher", role_description="Understands scientific logic."),
        Persona(name="Graduate Student", role_description="Learning biomedical reasoning."),
        Persona(name="Biostatician", role_description="Looking for data for secondary analysis"),
    ]

    # Build unified generator
    generator = TestsetGenerator(
        llm=llm,
        embedding_model=embeddings,
        knowledge_graph=kg,
        persona_list=personas,
    )

    print("[INFO] Generating testset...")
    testset = generator.generate(
        args.n_samples,
        query_distribution=[(MyMultiHopQuery(documents=nodes, kg=kg, llm=llm), 1.0)],
    )

    # Convert to DataFrame
    df = testset.to_pandas()
    import re

    # Extract PHS IDs from textual context RAGAS exports
    def extract_phs_ids_from_context(context_list):
        phs_ids = []
        if not isinstance(context_list, list):
            return phs_ids

        for block in context_list:
            # Extract "[PHS-ID: phsXXXX]"
            matches = re.findall(r"PHS-ID:\s*(phs[0-9\.v]+)", block)
            phs_ids.extend(matches)

        return list(sorted(set(phs_ids)))  # dedupe + sort

    df["phs_ids"] = df["reference_contexts"].apply(extract_phs_ids_from_context)

    # Split first two IDs into separate columns
    df["phsA"] = df["phs_ids"].apply(lambda x: x[0] if len(x) > 0 else None)
    df["phsB"] = df["phs_ids"].apply(lambda x: x[1] if len(x) > 1 else None)

    # Write final dataset
    outfile = Path(args.outdir) / "multihop_testset.jsonl"
    df.to_json(outfile, orient="records", lines=True)
    print(f"[DONE] Wrote {len(df)} examples → {outfile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--n-samples", type=int, default=40)
    args = parser.parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
