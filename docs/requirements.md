# Phase 0 requirements

## Product direction

SecondBrain AI is a personal knowledge management application intended to ingest a user's documents, answer questions using retrieval augmented generation (RAG), and later support agentic knowledge investigations.

## Implemented scope

- Establish a maintainable Python/FastAPI project foundation.
- Validate application configuration from environment variables.
- Expose a minimal health endpoint with injectable application logic.
- Add automated tests, linting, formatting, static type checking, and continuous integration.
- Document local setup and the initial architecture.
- Ingest PDF, Markdown, and UTF-8 plain-text files through a multipart API.
- Store original files on local disk and metadata plus extracted text in SQLite.
- List previously ingested document metadata.

## Current limitations and deferred scope

OCR, RAG, embeddings, vector storage, LangGraph, AWS/cloud storage, and authentication are not implemented. No credentials or secrets belong in source control.
