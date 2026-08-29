# @otisworks/curriculum-squad

A simple multi-agent pipeline runner. No framework, no magic - just agents, tasks, and results.

Give it a subject, get back a researched, source-verified course proposal: catalog description, learning outcomes, week-by-week schedule, reading list, assignments with rubrics, and a visual/media plan.

> **For developers:** setup and usage below are written for non-technical users. For the
> internals, skip to [How it works](#how-it-works) (architecture), [Providers](#providers)
> and [Search quality](#search-quality-differs-and-it-matters-here) (the provider abstraction
> and why search tiers aren't interchangeable), and [Customizing](#customizing) (extension
> points). One file, two dependencies, ESM, Node 18+. Importable as a module — `runTeam()`,
> `agents`, `tasks`, and `PROVIDERS` are all exported. `npm test` runs 62 checks and makes
> no API calls.

## The Pipeline

```
Archive (Research) → Dean (Outline) → Praxis (Materials) → Critic (Verify) → Palette (Visuals)
```

1. **Archive** - Subject Research Librarian. Maps the field: canon, critics, live debates, key practitioners, and how the subject is actually taught right now. *Has web search.*
2. **Dean** - Curriculum Architect. Turns research into a teachable course with measurable outcomes, a defensible arc, and a full week-by-week schedule.
3. **Praxis** - Assignment & Materials Designer. Builds the reading list, major assignments, rubrics, final project spec, and an honest workload audit.
4. **Critic** - Source Verifier & Curriculum Reviewer. Checks every citation actually exists, then audits outcome alignment, workload, sequencing, and coverage. *Has web search.*
5. **Palette** - Visual & Media Curator. Suggests images, diagrams, screenings, and artifacts that make abstract concepts concrete. *(optional)*

Each agent's output feeds into the next.

### Why the verification step matters

LLMs produce plausible-looking citations for books that do not exist. A reading list is exactly the wrong place for that. **Critic** searches for every cited work and labels it `VERIFIED`, `CORRECTED`, `UNVERIFIABLE`, or `LIKELY FABRICATED`.

Read that report before circulating anything. The pipeline reduces fabrication risk; it does not eliminate it.

## Requirements

- Node.js 18+
- An API key for one of: Anthropic, OpenAI, OpenRouter, DeepSeek

## Setup

```bash
npm install
cp .env.example .env
```

Then open `.env` in any text editor and add your API key. It's read automatically on every run.

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

`.env` is gitignored, so the key stays on your machine. If you'd rather not use a file, `export ANTHROPIC_API_KEY="..."` in your terminal still works and takes precedence.

### Subjects with spaces need quotes

```bash
node curriculum-squad.js "History of Cartography"     # correct
node curriculum-squad.js History of Cartography       # works; echoes back what it parsed
```

Unquoted arguments are joined into a single subject, and the result is echoed before the run starts. A misparse surfaces immediately rather than after a full pipeline.

## Providers

| Provider | Env var | Default model | Web search |
|---|---|---|---|
| `anthropic` *(default)* | `ANTHROPIC_API_KEY` | `claude-sonnet-5` | Native, agentic |
| `openai` | `OPENAI_API_KEY` | `gpt-5.6-terra` | Native, agentic |
| `openrouter` | `OPENROUTER_API_KEY` | `anthropic/claude-sonnet-5` | `web` plugin |
| `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-v4-pro` | **None** |

```bash
node curriculum-squad.js "Data Ethics" --provider=openrouter
LLM_PROVIDER=deepseek node curriculum-squad.js "Data Ethics"
```

OpenRouter and DeepSeek are OpenAI-compatible endpoints, so they reuse the OpenAI SDK with a different `baseURL`. Override any default model with the matching `*_MODEL` variable (`OPENROUTER_MODEL`, `DEEPSEEK_MODEL`, etc.).

### Search quality differs, and it matters here

The pipeline's credibility rests on Critic actually checking that cited books exist. The three search tiers are not equivalent:

**Native (Anthropic, OpenAI)** — the model searches repeatedly, on its own initiative, until satisfied. Best for verification, where each citation ideally gets its own lookup.

**OpenRouter `web` plugin** — grounds each request with a fixed set of results (default 5, tune with `WEB_PLUGIN_RESULTS`) before the model responds. It cannot search again after reading them. Fine for Archive's broad survey; genuinely thinner for Critic checking thirty citations individually. Sources retrieved are appended to the output so you can audit them. OpenRouter also offers a server-tool form of web search that restores agentic behaviour — a reasonable future upgrade.

**DeepSeek — none.** See below.

### Running without search

DeepSeek's API has no web search. Rather than pretend otherwise, the pipeline degrades explicitly:

- A loud startup warning.
- Archive is told to work from training data, prefer well-established works, and mark every citation `[UNVERIFIED]`.
- **Critic is forbidden from using the label `VERIFIED`.** Its status vocabulary is replaced with `PLAUSIBLE` / `SUSPECT` / `UNCHECKABLE`, and it must open its report by stating that nothing has been externally verified.
- `course-proposal.md` is stamped **"NO SOURCES IN THIS DOCUMENT HAVE BEEN VERIFIED"** at the top.

The reasoning: a model checking its own citations against its own memory proves nothing — that memory is what produced them. A confident `VERIFIED` in that situation is worse than silence, because it launders an unchecked list into one that looks audited.

Critic still does everything that doesn't need the web: outcome alignment, workload arithmetic, sequencing, rubric clarity, coverage. That part remains sound.

**Use DeepSeek for cheap structural drafts. Use a search-capable provider for anything you intend to submit.**

## Usage

```bash
# Bare subject, all defaults (15-week upper-division undergraduate seminar)
node curriculum-squad.js "Media Archaeology"

# Specify the course parameters
node curriculum-squad.js "Mycology for Artists" --level=grad --weeks=10 --format=studio

# Skip the visual/media plan
node curriculum-squad.js "Data Ethics" --weeks=12 --no-visuals

# Options
node curriculum-squad.js --help
```

### Options

| Flag | Values | Default |
|------|--------|---------|
| `--level` | `intro`, `undergrad`, `upper`, `grad`, `phd`, or any free text | `upper-division undergraduate` |
| `--weeks` | whole number, 1-52 | `15` |
| `--format` | `seminar`, `lecture`, `studio`, `lab`, `workshop`, `online`, or free text | `seminar` |
| `--provider` | `anthropic`, `openai`, `openrouter`, `deepseek` | `anthropic` |
| `--no-visuals` | skip the Palette agent | off |
| `--debug` | show full technical detail on failure | off |

Both `--flag=value` and `--flag value` work. The aliases are conveniences - any free-text value is passed straight through to the agents, so `--level="second-year MFA candidates"` is valid.

## Output

Saved to a timestamped folder in the current directory:

```
curriculum-2026-08-25T14-30-00-000Z/
├── course-proposal.md   # ← start here: outline + materials + visuals, stitched together
├── 1-research.md
├── 2-outline.md
├── 3-materials.md
├── 4-verification.md    # ← then read this: citation checks and committee critique
└── 5-visuals.md         # if not --no-visuals
```

Files are written **as each agent finishes**, so a failure partway through doesn't discard the work already paid for. `course-proposal.md` is assembled at the end with no extra API call.

## When something goes wrong

Errors are translated into plain language with a suggested fix — wrong key, unknown model, rate limit, out of credit, network blocked, response cut short. Add `--debug` for the raw technical detail.

Steps that fail transiently are **retried once automatically** before the run gives up, so a momentary provider hiccup doesn't discard fifteen minutes of work. Errors where a retry cannot help — bad API key, unknown model, billing — fail immediately.

Whatever finished before a failure is still written to disk, and the tool tells you where.

### Truncated output

Praxis produces the longest output in the pipeline — a full reading list plus assignments and rubrics — so it's the likeliest step to run out of room. A cut-off response is especially damaging here because it doesn't look broken: every downstream agent treats the fragment as a finished document, and Critic will "verify" a reading list that stops mid-table.

The run therefore aborts rather than continuing on a fragment when a response stops for length or returns implausibly short. If you hit this repeatedly, request a shorter course:

```bash
node curriculum-squad.js "your subject" --weeks=10
```

The thresholds: `MIN_OUTPUT_CHARS` (default 2000) is the minimum accepted response length; `CONFIG.maxTokens` (16384) is the output ceiling sent to the provider.

## Configuration

### Switch models

```bash
ANTHROPIC_MODEL=claude-opus-5 node curriculum-squad.js "Comparative Mythology"
OPENROUTER_MODEL=google/gemini-2.5-pro node curriculum-squad.js "X" --provider=openrouter
```

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `LLM_PROVIDER` | Provider to use | `anthropic` |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `OPENROUTER_API_KEY` / `DEEPSEEK_API_KEY` | Auth | — |
| `ANTHROPIC_MODEL` / `OPENAI_MODEL` / `OPENROUTER_MODEL` / `DEEPSEEK_MODEL` | Model override | per provider |
| `MAX_SEARCHES` | Search budget per agentic task | `30` |
| `MAX_SEARCH_TURNS` | Hard cap on agentic loop turns | `40` |
| `WEB_PLUGIN_RESULTS` | Results per OpenRouter grounding pass | `5` |
| `MIN_OUTPUT_CHARS` | Truncation floor | `2000` |

## How it works

It's just functions. No classes, no framework magic.

- `agents` - Object defining each agent's name, role, goal, and background
- `tasks` - Array of tasks whose prompts are functions of `(inputs, results, ctx)`, so each can read every prior agent's output and adapt to whether search is available
- `PROVIDERS` - Registry of provider endpoints, models, and search capability
- `callLLM()` - Handles API calls, routing by provider and search style
- `runTeam()` - Executes tasks sequentially, passing results forward

Adding an agent means adding one entry to `agents` and one to `tasks`. Nothing else needs to change.

## Customizing

The prompts are the product - they're plain template strings, edit them freely.

- Want a 5-agent → 4-agent pipeline? Delete a task and its agent.
- Teaching in a specific department? Add its conventions to `Dean`'s task prompt.
- Need accreditation standards? Add them to `Archive`'s research prompt.
- `CITATION_RULE` is shared by every source-producing agent - one edit hits them all.

## Future ideas

- [ ] Resume from checkpoint if a task fails
- [ ] Streaming output during long tasks
- [ ] Batch mode for multiple subjects (integrate with batch-kit)
- [ ] Config files for custom agents/pipelines
- [ ] Export to LMS-ready formats (Canvas, Moodle)

## License

MIT
