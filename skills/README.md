# Skills

Reusable **operator prompts** for AI agents working with RAGrig. Each `.md` file
is a self-contained skill: a structured set of instructions an AI agent follows
to complete a specific workflow.

## Convention

Skills use YAML frontmatter (between `---` delimiters) for metadata, followed by
Markdown instruction content. Any agent that understands YAML frontmatter can
parse the metadata; any agent can read the Markdown instructions directly.

Frontmatter fields:

| Field | Purpose |
|-------|---------|
| `name` | Stable identifier — matches the filename stem |
| `version` | Semantic version string |
| `description` | One-line description used for intent matching and tool UIs |
| `scope` | `project` (RAGrig-specific) or `global` |
| `apply` | `on-demand` (invoked explicitly) or `always` (injected into every context) |
| `triggers` | Natural-language phrases that should activate this skill |
| `arguments` | Named parameters the agent should collect before executing |
| `prerequisites` | Conditions that must be true before the skill runs |
| `tools_required` | Tool capabilities needed (bash, http, read_file, …) |
| `compatible_with` | Informational: which agent tools are known to work with this skill |

## Standard workflow (follow in order)

1. **Create a config file** — required before setup or evaluation
2. **Set up the worker** — start, load models, build collection
3. **Run evaluation** — pre-flight checks, execute, interpret
4. **Analyze results** — export traces, explore metrics

## Available skills

| Step | File | Workflow |
|------|------|---------|
| 1 | [`ragrig-config.md`](ragrig-config.md) | Interactively create a config YAML (required first step) |
| 2 | [`ragrig-setup.md`](ragrig-setup.md) | Bootstrap venv, start worker, load GGUF models, build ChromaDB collection |
| 3 | [`ragrig-evaluate.md`](ragrig-evaluate.md) | Pre-flight checks, run evaluation via CLI or HTTP API, interpret results |
| 4 | [`ragrig-analyze.md`](ragrig-analyze.md) | Export Langfuse traces to Parquet, explore metrics in notebooks |
| — | [`ragrig-troubleshoot.md`](ragrig-troubleshoot.md) | Diagnose failures, consult failure mode table, verify env vars |

> **Config first.** `ragrig-setup` and `ragrig-evaluate` both require a config
> file. If none exists, they will redirect to `ragrig-config` automatically.
> The full config reference is in [`docs/config.md`](../docs/config.md).

## Tool adapters

Thin wrapper files that point agent tools at the canonical skill files above:

| Tool | Adapter location | Invocation |
|------|-----------------|-----------|
| Claude Code | `.claude/commands/ragrig*.md` | `/ragrig`, `/ragrig-setup`, `/ragrig-evaluate`, `/ragrig-analyze`, `/ragrig-troubleshoot` |
| Cursor | `.cursor/rules/ragrig.mdc` | Attached automatically when config YAML or `.env` file is open |
| GitHub Copilot | `.github/copilot-instructions.md` | Always-on context |
| OpenAI Codex / generic agents | `AGENTS.md` (project root) | Points here |

## Adding a new skill

1. Create `skills/<skill-name>.md` with the YAML frontmatter schema above
2. Add a row to the table in this README
3. Add a Claude Code adapter in `.claude/commands/<skill-name>.md`
4. Update the Cursor and Copilot adapter tables
