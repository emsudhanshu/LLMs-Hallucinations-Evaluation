# Medical QA Hallucination Analysis

This project now uses a simpler stack built around LangGraph, FAISS, and Ollama.

- `no_rag`
- `rag`

It measures:

- accuracy
- error rate
- hallucination categories

## Simplified stack

```text
LangGraph orchestration
FAISS vector store
Ollama embeddings: nomic-embed-text
Ollama generation: llama3:latest
Knowledge base JSON: data/knowledge_base.json
```

## Config

Model selection is controlled from:

- [config.json](/Users/sudhanshukakkar/Desktop/semester%203/NLP/project/Code/LLMs-Hallucinations-Evaluation/config.json:1)

For now the defaults are:

- `answer_agent`
- `verifier_agent`
- `retrieval.embedding_model`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional `.env`:

```bash
OPENAI_API_KEY=...
GEMINI_API_KEY=...
OLLAMA_BASE_URL=http://localhost:11434
```

## Run

Default no-RAG:

```bash
python3 main.py --mode no_rag
```

Default RAG:

```bash
python3 main.py --mode rag
```

Rebuild the knowledge base / FAISS index:

```bash
python3 main.py --mode rag --rebuild-kb
```

## Outputs

- `results/no_rag_results.csv`
- `results/rag_results.csv`
- `results/no_rag_verified_results.csv`
- `results/rag_verified_results.csv`
- `data/knowledge_base.json`
- `artifacts/faiss_kb/`
