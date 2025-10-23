from __future__ import annotations
import json, re, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

try:
    import spacy
    _HAS_SPACY = True
except Exception:
    _HAS_SPACY = False

import networkx as nx

# 1) Loads abstract-like studies from a JSON file (StudyId, StudyName, Description).
# 2) Runs lightweight biomedical NER (scispaCy if available; else conservative fallback).
# 3) Detects relation cue words in sentences (treats/causes/inhibits/...).
# 4) Builds a NetworkX MultiDiGraph: nodes have {name, type}, edges have {relation, evidence, confidence}.
# 5) Verbalizes local subgraphs around center nodes into readable sentences.
# 6) Emits:
#    - kg_edges.jsonl    
#    - kg_contexts.jsonl (
#
# Notes:
# - RAGAS consumes TEXT, not graphs. These “contexts” are the textual bridge.
# - Relation detection is pattern-based and approximate (no negation handling).
# - The fallback NER defaults unknown capitalized spans to Disease (very weak prior).

# Label sets for scispaCy models

CHEM_LABELS = {"CHEMICAL", "CHEMICAL_ENTITY", "CHEMICALSUBSTANCE"}
DISEASE_LABELS = {"DISEASE", "DISEASE_OR_SYNDROME"}
GENE_LABELS = {"GENE_OR_GENE_PRODUCT", "GENE", "PROTEIN"}

REL_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(treats?|therapy for|effective against|ameliorates)\b", "treats"),
    (r"\b(causes?|leads to|induces?|triggers?)\b", "causes"),
    (r"\b(inhibits?|antagonizes?|blocks)\b", "inhibits"),
    (r"\b(activates?|upregulates?)\b", "activates"),
    (r"\b(associated with|linked to|correlates with)\b", "associated_with"),
]
REL_PATTERNS = [(re.compile(p, re.I), r) for p, r in REL_PATTERNS]

import pandas as pd  # add this import at the top if not already there

def load_abstracts(path: str | Path) -> List[Dict[str, Any]]:
    """
    Load dataset of abstracts (StudyId, StudyName, Description).
    Supports both JSON and CSV.
    """
    path = str(path)
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        data = df.to_dict(orient="records")
    else:
        data = json.loads(Path(path).read_text())

    items = []
    for row in data:
        items.append({
            "doc_id": row.get("StudyId") or row.get("id") or row.get("DocId"),
            "title": row.get("StudyName") or row.get("title") or "",
            "abstract": row.get("Description") or row.get("abstract") or "",
            "permalink": row.get("Permalink") or "",
        })
    return items

# def load_abstracts(path: str | Path) -> List[Dict[str, Any]]:
#     """
#     Expects a JSON array with objects containing at least:
#       - StudyId
#       - StudyName
#       - Description (we'll treat as 'abstract')
#     """
#     with open(path, "r") as f:
#         data = json.load(f)
#     # Normalize a minimal schema
#     items = []
#     for row in data:
#         items.append({
#             "doc_id": row.get("StudyId") or row.get("id") or row.get("DocId"),
#             "title": row.get("StudyName") or row.get("title") or "",
#             "abstract": row.get("Description") or row.get("abstract") or "",
#             "permalink": row.get("Permalink") or "",
#         })
#     return items


_SENT_SPLIT = re.compile(r"(?<=[\.!?])\s+")

def split_sentences(text: str) -> List[str]:
    text = text.strip().replace("\n", " ")
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return sents


# NER (scispaCy optional)
def _load_pipes() -> Tuple[Any | None, Any | None]:
    """
    Try to load scispaCy models; return (nlp_cdr, nlp_bio) or (None, None).
    """
    if not _HAS_SPACY:
        return None, None
    try:
        nlp_cdr = spacy.load("en_ner_bc5cdr_md")      # chemicals/diseases
    except Exception:
        nlp_cdr = None
    try:
        nlp_bio = spacy.load("en_ner_bionlp13cg_md")  # gene/protein etc.
    except Exception:
        nlp_bio = None
    return nlp_cdr, nlp_bio


def extract_entities(sent: str, nlp_cdr=None, nlp_bio=None) -> List[Dict[str, Any]]:
    """
    Return [{text, type}] with type in {Chemical, Disease, GeneOrProtein}
    Falls back to weak regex if scispaCy models are unavailable.
    """
    out: List[Dict[str, Any]] = []
    if nlp_cdr:
        for e in nlp_cdr(sent).ents:
            lab = e.label_.upper()
            if lab in CHEM_LABELS:
                out.append({"text": e.text, "type": "Chemical"})
            elif lab in DISEASE_LABELS:
                out.append({"text": e.text, "type": "Disease"})
    if nlp_bio:
        for e in nlp_bio(sent).ents:
            lab = e.label_.upper()
            if lab in GENE_LABELS:
                out.append({"text": e.text, "type": "GeneOrProtein"})

    if not out:  # very light fallback via capitalization + biomedical-ish tokens
        # Look for capitalized multiword terms as a weak proxy
        caps = re.findall(r"\b([A-Z][a-zA-Z0-9\-]{2,}(?:\s+[A-Z][a-zA-Z0-9\-]{2,})*)\b", sent)
        for c in caps[:3]:  # keep it conservative
            out.append({"text": c, "type": "Disease"})  # default to Disease as weakest prior

    # de-dup within sentence
    seen = set()
    uniq = []
    for e in out:
        key = (e["type"], e["text"].lower())
        if key not in seen:
            seen.add(key)
            uniq.append(e)
    return uniq


# 4) Relation cues from text
def detect_relations(sent: str) -> List[str]:
    labels = [lab for pat, lab in REL_PATTERNS if pat.search(sent)]
    return labels or ["associated_with"]


# 5) Graph building
def _node_id(etype: str, name: str) -> str:
    return f"{etype}::{name.lower()}"

def build_kg(rows: List[Dict[str, Any]]) -> nx.MultiDiGraph:
    """
    Build a MultiDiGraph with nodes {name, type} and edges {relation, evidence, confidence}
    """
    G = nx.MultiDiGraph()
    nlp_cdr, nlp_bio = _load_pipes()

    for row in rows:
        doc_id = row["doc_id"]
        abstract = (row.get("title", "") + ". " + row.get("abstract", "")).strip()
        if not abstract:
            continue

        for sent in split_sentences(abstract):
            ents = extract_entities(sent, nlp_cdr, nlp_bio)
            if len(ents) < 2:
                continue

            rels = detect_relations(sent)
            # add nodes
            for e in ents:
                nid = _node_id(e["type"], e["text"])
                G.add_node(nid, name=e["text"], type=e["type"])

            # fully connect co-mentioned entities in sentence
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    a, b = ents[i], ents[j]
                    u, v = _node_id(a["type"], a["text"]), _node_id(b["type"], b["text"])
                    for r in rels:
                        G.add_edge(
                            u, v,
                            relation=r,
                            evidence={"doc_id": doc_id, "sentence": sent},
                            confidence=0.6 if r != "associated_with" else 0.4,
                        )
    return G


# Verbalization for RAGAS
def verbalize_edge(u: str, v: str, ed: Dict[str, Any], G: nx.MultiDiGraph) -> str:
    su, sv = G.nodes[u], G.nodes[v]
    r = ed.get("relation", "associated_with")
    u_name, v_name = su.get("name", "X"), sv.get("name", "Y")
    if r == "treats":
        return f"{u_name} is used to treat {v_name}."
    if r == "causes":
        return f"{u_name} can cause {v_name}."
    if r == "inhibits":
        return f"{u_name} inhibits {v_name}."
    if r == "activates":
        return f"{u_name} activates {v_name}."
    return f"{u_name} is associated with {v_name}."

def subgraph_context(G: nx.MultiDiGraph, center: str, hops: int = 1, max_edges: int = 12) -> str:
    # BFS frontier
    nodes = {center}
    frontier = {center}
    for _ in range(hops):
        newf = set()
        for n in list(frontier):
            newf |= set(G.predecessors(n)) | set(G.successors(n))
        frontier = newf - nodes
        nodes |= frontier

    sents = []
    for u, v, ed in G.edges(data=True):
        if u in nodes or v in nodes:
            sents.append(verbalize_edge(u, v, ed, G))
            if len(sents) >= max_edges:
                break
    # dedupe while preserving order
    sents = list(dict.fromkeys(sents))
    return " ".join(sents)


def make_contexts(G: nx.MultiDiGraph, hops: int = 1, max_edges: int = 12, max_centers: int = 200) -> List[Dict[str, Any]]:
    # choose centers by degree; prefer Disease -> Chemical -> GeneOrProtein
    nodes_by_type = defaultdict(list)
    for n, d in G.nodes(data=True):
        nodes_by_type[d.get("type", "Other")].append(n)

    def topk(ns):  # degree-based sorting
        return sorted(ns, key=lambda x: G.degree(x), reverse=True)

    centers = []
    centers += topk(nodes_by_type.get("Disease", []))
    centers += topk(nodes_by_type.get("Chemical", []))
    centers += topk(nodes_by_type.get("GeneOrProtein", []))
    centers = centers[:max_centers]

    out = []
    for cid in centers:
        ctx = subgraph_context(G, cid, hops=hops, max_edges=max_edges)
        if ctx.strip():
            out.append({
                "center_id": cid,
                "center_name": G.nodes[cid].get("name", ""),
                "center_type": G.nodes[cid].get("type", ""),
                "context": ctx
            })
    # dedupe contexts
    seen = set()
    uniq = []
    for r in out:
        if r["context"] not in seen:
            seen.add(r["context"])
            uniq.append(r)
    return uniq


# 7) Persistence
def save_edges_jsonl(G: nx.MultiDiGraph, outpath: str | Path):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        for u, v, ed in G.edges(data=True):
            rec = {
                "src": u, "src_name": G.nodes[u].get("name"), "src_type": G.nodes[u].get("type"),
                "tgt": v, "tgt_name": G.nodes[v].get("name"), "tgt_type": G.nodes[v].get("type"),
                "relation": ed.get("relation"),
                "evidence": ed.get("evidence"),
                "confidence": ed.get("confidence", 0.5),
            }
            f.write(json.dumps(rec) + "\n")


def save_contexts_jsonl(contexts: List[Dict[str, Any]], outpath: str | Path):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        for r in contexts:
            f.write(json.dumps(r) + "\n")


# CLI
def main():
    p = argparse.ArgumentParser(description="Build a lightweight KG from abstracts and emit contexts for RAGAS.")
    p.add_argument("--input", required=True, help="JSON file with abstracts (array of objects).")
    p.add_argument("--outdir", required=True, help="Directory for outputs.")
    p.add_argument("--hops", type=int, default=1, help="Subgraph hops for context verbalization.")
    p.add_argument("--max-edges", type=int, default=12, help="Max edges per context.")
    p.add_argument("--max-centers", type=int, default=200, help="Limit number of centers.")
    args = p.parse_args()

    rows = load_abstracts(args.input)
    G = build_kg(rows)
    contexts = make_contexts(G, hops=args.hops, max_edges=args.max_edges, max_centers=args.max_centers)

    outdir = Path(args.outdir)
    save_edges_jsonl(G, outdir / "kg_edges.jsonl")
    save_contexts_jsonl(contexts, outdir / "kg_contexts.jsonl")

    print(f"[KG] nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    print(f"[CTX] contexts={len(contexts)} → {outdir/'kg_contexts.jsonl'}")

if __name__ == "__main__":
    main()
