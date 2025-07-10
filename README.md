
# DugBot - Biomedical Research Study Discovery Assistant

DugBot is an advanced AI-powered research assistant designed to help researchers discover and understand biomedical study through natural language queries. Built on LangGraph and LangChain frameworks, it provides intelligent search capabilities over the NHLBI-BioData Catalyst research studies using both vector-based similarity search and knowledge graph traversal.

## Overview

DugBot employs Retrieval Augmented Generation (RAG) to provide accurate, grounded responses about biomedical studies, variables, and their relationships. The system processes 200+ studies from the BioData Catalyst database and enables researchers to find relevant information through conversational queries.

## Architecture

### Multi-Agent System
- **Intent Agent**: Analyzes user queries and extracts intent/preferences
- **Supervisor Agent**: Routes queries to appropriate lookup mechanisms
- **KG Lookup Agent**: Retrieves information through knowledge graph traversal
- **QV Lookup Agent**: Performs vector-based similarity search

### Dual Retrieval Mechanisms
1. **Vector-Based Retrieval**: Semantic similarity search over pre-generated questions
2. **Knowledge Graph Retrieval**: Concept-based graph traversal for entity relationships

<img width="2000" height="808" alt="Copy of Dug Bot - Combined Lookup" src="https://github.com/user-attachments/assets/b6ee5e63-54c0-4a6a-84c5-b0bd16521ddd" />

## Getting Started

### Prerequisites
- Python 3.8+
- Docker (optional, for containerized deployment)
- Ollama (for local LLM inference)
- Qdrant (vector database)
- Redis (knowledge graph storage)

## Usage

### Server Deployment

The system provides multiple server endpoints for different use cases:

1. **Agent Server** (Multi-agent routing)
```bash
python src/agent_server.py
# Serves on port 8099
```

2. **Knowledge Graph Server** (KG-only queries)
```bash
python src/kg_app_server.py
# Serves on port 8094
```

3. **Combined QVKG Server** (Vector + KG)
```bash
python src/qvkg_app_server.py
# Serves on port 8005
```

### API Endpoints

#### Agent Server
- `POST /agent/invoke` - Full agent routing with intent analysis
- `POST /agent/stream` - Streaming responses
- `GET /agent/score/{trace_id}/{score}` - Feedback scoring

#### Knowledge Graph Server
- `POST /kg-app/invoke` - Knowledge graph-based queries
- `POST /kg-app/stream` - Streaming KG responses

#### Combined Server
- `POST /qvkg-app/invoke` - Combined vector + KG queries
- `POST /qvkg-app/stream` - Streaming combined responses

## Data Processing

### Question Generation
The system includes tools for generating training questions from study abstracts:

```bash
# Generate questions from study abstracts
python src/core.py
```

### Database Population
```bash
# Create embeddings for question-answer pairs
python db_builder/create_embeddings.py

# Load data into Qdrant
python db_builder/qdrant_loader.py
```

## Project Structure

```
Koios-develop/
├── src/
│   ├── agents/                 # Multi-agent implementations
│   │   ├── intent_agent_graph.py
│   │   ├── route_agentic_graph.py
│   │   ├── combined_context_graph.py
│   │   └── supervisor.py
│   ├── chains/                 # Processing chains
│   │   ├── kg_chain.py
│   │   ├── question_lookup_chain.py
│   │   ├── qvkg_chain.py
│   │   └── user_intent_chain.py
│   ├── models/                 # Data models
│   │   ├── agent_state.py
│   │   └── user_question.py
│   ├── databases/              # Database connectors
│   │   ├── qdrant.py
│   │   └── redis_graph.py
│   ├── guardrails/             # Input validation
│   │   └── input_guard.py
│   └── util/                   # Utility functions
├── db_builder/                 # Database setup tools
├── prompts/                    # Prompt templates
├── ragas_benchmark/            # Evaluation framework
└── requirements.txt
```

## License

This project is licensed under the MIT License 
