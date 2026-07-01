---
title: SHL Assistant
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# SHL Assessment Recommender API

A stateless conversational agent built using FastAPI that recommends SHL assessments from the official catalog.

## Technical Details

- **Framework:** FastAPI
- **Search System:** Hybrid retrieval combining FAISS vector search and BM25 term matching.
- **LLM Integration:** Groq (Llama-3.1-8b-instant) as primary, with a fallback to Google Gemini (gemini-3.1-flash-lite).
- **Validation Layer:** Custom regex-based normalization layer to match and validate all recommendations against canonical catalog items, completely eliminating link and name hallucinations.

## API Endpoints

- `GET /health`: Health check return `{"status": "ok"}`
- `POST /chat`: Stateful conversation trace processing
