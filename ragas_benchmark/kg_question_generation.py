"""
Generate evaluation questions from KG contexts using RAGAS' TestsetGenerator.
Defaults to local vLLM + Ollama; uses OpenAI only if OPENAI_API_KEY is set.
Coalesces short KG contexts into longer Documents to satisfy RAGAS' min length.
"""

import argparse, json, os
from pathlib import Path
from collections import defaultdict

from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.graph import Node, Relationship, KnowledgeGraph, NodeType

try:
    from langchain_core.documents import Document  # LC >=0.2
except Exception:
    from langchain.schema import Document          # LC <0.2

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings


# This script loads the text "contexts" produced from the abstract-derived KG,
# merges short snippets into longer Documents (RAGAS prefers >= ~100 tokens),
# and uses RAGAS' TestsetGenerator to synthesize Q–A pairs for evaluation.
#
# Typical use:
#   python ragas_gen.py --contexts out/kg_contexts.jsonl --out out/ragas_testset.jsonl --n 3
#
# Notes:
# - RAGAS consumes TEXT documents (LangChain Document objects), not graphs.
# - If contexts are short, coalescing is essential to pass RAGAS' token threshold.
# - By default this script tries OpenAI if OPENAI_API_KEY is set; otherwise it uses local Ollama.

from itertools import combinations

async def _generate_scenarios(self, n: int, knowledge_graph: KnowledgeGraph, callbacks=None):
    # Build adjacency map
    adj = defaultdict(set)
    for rel in knowledge_graph.relationships:
        if rel.source.id != rel.target.id:
            adj[rel.source.id].add(rel.target.id)
            adj[rel.target.id].add(rel.source.id)

    node_by_id = {n.id: n for n in knowledge_graph.nodes}
    scenarios = []
    seen = set()

    for node_id, neighbors in adj.items():
        # create small connected subgraphs (3–5 nodes)
        cluster = [node_id] + list(neighbors)
        cluster = cluster[:5]  # limit size
        key = tuple(sorted(cluster))
        if key in seen or len(cluster) < 3:
            continue
        seen.add(key)

        # compute an average similarity score across edges in cluster
        scores = []
        for a, b in combinations(cluster, 2):
            rels = [
                r for r in knowledge_graph.relationships
                if {r.source.id, r.target.id} == {a, b}
            ]
            for r in rels:
                props = r.properties or {}
                s = max(
                    float(props.get("entity_jaccard_similarity", 0.0)),
                    float(props.get("keyphrase_jaccard_similarity", 0.0))
                )
                if s > 0:
                    scores.append(s)
        score = sum(scores) / len(scores) if scores else 0

        scenarios.append({
            "type": "multi",
            "nodes": cluster,
            "score": score,
        })

        if len(scenarios) >= n:
            break

    return scenarios


def _approx_tokens(text: str) -> int:
    # quick/loose token estimate good enough for thresholding
    return max(0, len(text.split()))

def load_contexts(path: str | Path):
    docs = []
    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line)
            ctx = (rec.get("context") or "").strip()
            if not ctx:
                continue
            docs.append(Document(page_content=ctx, metadata=rec))
    return docs

def coalesce_docs(docs, min_tokens: int = 120, group_key: str = "StudyId"):
    """
    Merge short context snippets into longer Documents.
    Priority: group by StudyId (if present) then by StudyName; if absent, batch sequentially.
    """
    grouped = defaultdict(list)
    for d in docs:
        meta = d.metadata or {}
        key = meta.get(group_key) or meta.get("StudyName") or "__ungrouped__"
        grouped[key].append(d.page_content)

    merged_docs = []
    for key, snippets in grouped.items():
        buf = []
        acc = 0
        for s in snippets:
            buf.append(s)
            acc += _approx_tokens(s)
            if acc >= min_tokens:
                merged_docs.append(Document(page_content="\n\n".join(buf), metadata={"group_key": key, "count": len(buf)}))
                buf, acc = [], 0
        # flush remainder if it’s reasonably long
        if acc >= min_tokens:
            merged_docs.append(Document(page_content="\n\n".join(buf), metadata={"group_key": key, "count": len(buf)}))
        # else drop tiny tail; they’ll just be ignored to satisfy RAGAS min length
    return merged_docs

def get_llm_and_embeddings():
    if os.getenv("OPENAI_API_KEY"):
        print("[INFO] Using OpenAI backend")
        llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
        emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    else:
        print("[INFO] Using local Ollama backend")
        llm = LangchainLLMWrapper(
            ChatOllama(
                model=os.getenv("GEN_MODEL_NAME", "llama3.1:latest"),
                base_url=os.getenv("LLM_URL", "http://localhost:11434"),
                temperature=float(os.getenv("GEN_TEMPERATURE", "0")),
            )
        )
        emb = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(
                model=os.getenv("EMB_MODEL_NAME", "nomic-embed-text"),
                base_url=os.getenv("EMBEDDING_URL", "http://localhost:11434"),
            )
        )
    return llm, emb

def generate_testset(docs, outpath: str | Path, testset_size: int = 30):
    # Coalesce short docs first
    long_docs = coalesce_docs(docs, min_tokens=140)  # a tad above 100 for safety
    if not long_docs:
        raise SystemExit("No documents long enough to pass RAGAS min token threshold. "
                         "Increase KG context size or reduce min_tokens in coalesce_docs().")
    print(f"[INFO] Using {len(long_docs)} merged Documents (from {len(docs)} snippets).")

    llm, emb = get_llm_and_embeddings()
    gen = TestsetGenerator(llm=llm, embedding_model=emb)

    testset = gen.generate_with_langchain_docs(long_docs, testset_size=testset_size)

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    testset.to_jsonl(outpath)
    print(f"[RAGAS] wrote {len(testset)} samples → {outpath}")

def main():
    ap = argparse.ArgumentParser(description="Generate questions from KG contexts using RAGAS (0.3.x)")
    ap.add_argument("--contexts", required=True, help="kg_contexts.jsonl produced by kg_builder.py")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--n", type=int, default=3, help="~questions per context (approx)")
    args = ap.parse_args()

    docs = load_contexts(args.contexts)
    # target roughly n questions per original context, but doc merging reduces count.
    testset_size = max(1, args.n) * max(1, len(docs))
    generate_testset(docs, args.out, testset_size=testset_size)

if __name__ == "__main__":
    main()
