
# Medical RAG — Intelligent Medical Nomenclature Assistant

An intelligent Retrieval-Augmented Generation (RAG) system designed to retrieve, analyze and explain medical nomenclature information using Natural Language Processing, Machine Learning and Large Language Models.

## Overview

This project aims to build a custom RAG architecture capable of understanding user queries and retrieving relevant medical acts from structured medical nomenclature documents.

Unlike a basic semantic search system, the planned architecture integrates an intelligent query classification mechanism to select the most appropriate retrieval strategy.

The system is being developed from scratch, without relying on high-level RAG frameworks such as LangChain or LlamaIndex.

## Architecture

The project follows a modular architecture:

```text
src/
├── ingestion/       # Document loading and extraction
├── preprocessing/   # Text cleaning and normalization
├── chunking/        # Document segmentation
├── embeddings/      # Text vectorization
├── vectordb/        # ChromaDB integration
├── ml/              # Machine Learning modules
├── retrieval/       # Information retrieval strategies
├── prompts/         # Prompt templates
├── llm/             # LLM integration
├── api/             # Future API layer
└── utils/           # Configuration and logging
```

## Technologies

- **Python 3.11**
- **NLP:** Text preprocessing and semantic representation
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Database:** ChromaDB
- **Machine Learning:** Scikit-learn
- **LLM:** Ollama / Mistral
- **Document Processing:** Python-docx
- **Version Control:** Git / GitHub

## Implemented Components

| Component | Description | Status |
|---|---|---|
| Document Ingestion | DOCX extraction and structured metadata | Completed |
| Preprocessing | Text cleaning and normalization | Completed |
| Chunking | Medical act segmentation | Completed |
| Embeddings | Vector generation (384 dimensions) | Completed |
| Vector Database | ChromaDB persistence and insertion | Completed |
| Configuration | Centralized project configuration | Completed |
| Logging | Centralized logging system | Completed |
| Query Classification | ML-based query intent classification | In progress |
| Retrieval | Lexical, semantic and code-based search | Planned |
| LLM Generation | Context-grounded response generation | Planned |

## Intelligent Retrieval Strategy

The planned system will classify user queries and route them to an appropriate retrieval method:

| Query Type | Retrieval Strategy |
|---|---|
| Keyword-based query | Lexical search |
| Natural language query | Semantic search |
| Medical act code | Exact metadata search |

This hybrid approach aims to improve retrieval relevance and reduce unnecessary semantic searches.

## Current Dataset

The current ingestion pipeline extracts structured medical acts from DOCX nomenclature documents.

Current processing results:

- 736 extracted medical acts
- 736 generated chunks
- 384-dimensional embeddings
- 736 records stored in ChromaDB

*Source documents and the local vector database are not included in this repository.*

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Projet_rag_ollama
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Ollama

Install Ollama and download the Mistral model:

```bash
ollama pull mistral
```

Make sure the Ollama service is running before using the LLM component.

## Project Configuration

Project parameters are centralized in:

```text
src/utils/config.py
```

This file contains:

- Embedding model configuration
- Embedding dimensions and batch size
- ChromaDB storage path and collection
- Ollama URL and model name
- Retrieval parameters

## Development Roadmap

- [x] DOCX ingestion pipeline
- [x] Text preprocessing
- [x] Medical document chunking
- [x] Embedding generation
- [x] ChromaDB integration
- [x] Centralized configuration
- [x] Logging
- [ ] Query intent classification integration
- [ ] Hybrid retrieval engine
- [ ] Context selection and ranking
- [ ] Ollama / Mistral integration
- [ ] Grounded answer generation
- [ ] API development
- [ ] Evaluation and optimization

## Design Principles

- Modular and maintainable architecture
- Local-first processing
- Explicit control over each RAG component
- No dependency on high-level RAG frameworks
- Separation between retrieval and generation

## Author

AI & Machine Learning Engineering Student

**Focus:** Artificial Intelligence, NLP, Machine Learning and Data Engineering.
