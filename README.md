# ApplyWise

ApplyWise is an AI agent that prepares a job application from a role description and resume. It uses the OpenAI Responses API with function calling, searches resume lines locally, asks the model to compare requirements with returned evidence, and stages an editable application draft for review.

## Run it locally

Requirements: Python 3.10+ and an OpenAI API key with API access and billing enabled.

```bash
python -m applywise.app
```

Open <http://127.0.0.1:8000>. Paste the job description and resume, enter your API key, keep or change the model ID, and check the consent box. The default model is `gpt-6.1-terra`. The run makes multiple Responses API calls while the model uses tools. API usage is billed to the key's account; check the current model price in your API account before running.

## What the agent does

1. Sends the job description and resume to the selected OpenAI model after the consent checkbox is checked.
2. The model chooses `search_resume_evidence` calls. The app searches resume lines on the local machine and returns exact snippets with evidence IDs.
3. The model uses those tool results to prepare requirements, evidence-backed matches, gaps, an action plan, a cover letter, and evidence references for factual claims.
4. `stage_application_draft` checks that all cited evidence IDs actually came from local search and holds the draft in memory.
5. The person reviews the content and explicitly approves a local text download. The app never applies to jobs or sends the draft to anyone.

The model can still make a semantic mismatch when connecting a claim to a cited line. Always verify the cited excerpt supports the claim and edit the draft before using it. The local search uses word overlap and can miss valid paraphrases or synonyms; a missing match does not mean a missing skill.

## Privacy and security behavior

- The default web server binds to `127.0.0.1` only.
- ApplyWise does not write the key, resume, job text, or draft to disk and suppresses request-body logging. Staged drafts expire from process memory after 45 minutes or when the server stops.
- The API key is sent from the browser to the app server for the current request and is not placed back into the page. The server operator can technically access it while processing. The server then uses HTTPS to send API requests to `api.openai.com`.
- Requests set `store: false`; review your provider account and data controls for applicable processing terms.
- The consent checkbox must be selected before the job, resume, and key pass through the app server to the API.
- A human must check a separate approval box before the local cover-letter download is unlocked.
- Same-origin checks, bounded request sizes, security headers, and concurrent request handling are enabled.

## Hosting for other users

Users bring their own API key; ApplyWise has no shared key or shared API bill. A Dockerfile is included for a container host. To build and try the container locally:

```bash
docker build -t applywise .
docker run --rm -p 8000:8000 applywise
```

The container listens on `0.0.0.0`; a public deployment must use HTTPS. Before sharing a hosted instance, choose a host you trust: each user's API key and resume pass through its server while a request is processed. ApplyWise does not save them, but the host operator and infrastructure provider may still be able to inspect data in transit or at runtime. This starter does not provide user accounts, persistent storage, or production monitoring. Do not publish it on an untrusted host.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests mock the Responses API and exercise the complete tool loop, evidence-ID validation, the approval gate, API-key non-persistence in the page, same-origin checks, HTML escaping, and input validation. They do not make a live API call.

## Main files

- `applywise/agent.py` — Responses API client, tool schemas, local evidence search, agent loop, draft validation, approval/export gate.
- `applywise/app.py` — local web interface and in-memory session handling.
- `tests/test_agent.py` — mocked agent-flow tests.
