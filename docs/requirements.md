# Phase 0 requirements

## Product direction

SecondBrain AI is a personal knowledge management application intended to ingest a user's documents, answer questions using retrieval augmented generation (RAG), and later support agentic knowledge investigations.

## Session 1 scope

- Establish a maintainable Python/FastAPI project foundation.
- Validate application configuration from environment variables.
- Expose a minimal health endpoint with injectable application logic.
- Add automated tests, linting, formatting, static type checking, and continuous integration.
- Document local setup and the initial architecture.

## Out of scope

Document ingestion, RAG, embeddings, vector storage, LangGraph, AWS integrations, and authentication are deferred to later sessions. No credentials or secrets belong in source control.
