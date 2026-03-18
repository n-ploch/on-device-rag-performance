# GitHub Copilot Instructions — RAGrig

For project architecture, setup commands, API reference, and env vars: see `AGENTS.md`.

## Available skills

Reusable operator prompts are in `skills/`. When a user asks you to perform
a RAGrig workflow, load the appropriate skill file and follow it exactly:

| User intent | Skill file |
|-------------|-----------|
| Set up / bootstrap / load models / build collection | `skills/ragrig-setup.md` |
| Run an evaluation / benchmark | `skills/ragrig-evaluate.md` |
| Export / analyze results / explore metrics | `skills/ragrig-analyze.md` |
| Diagnose a failure / something not working | `skills/ragrig-troubleshoot.md` |
