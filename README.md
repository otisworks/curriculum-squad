# @otisworks/curriculum-squad

A simple multi-agent pipeline runner. No framework, no magic - just agents, tasks, and results.

Give it a subject, get back a researched, source-verified course proposal: catalog description, learning outcomes, week-by-week schedule, reading list, assignments with rubrics, and a visual/media plan.

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
- Anthropic API key (or OpenAI)

## Setup

```bash
npm install
export ANTHROPIC_API_KEY="sk-ant-..."
```

> Note: there is no `.env` support - export the variable in your shell, or prefix the command.

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
| `--no-visuals` | skip the Palette agent | off |

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

## Configuration

### Switch models

```bash
ANTHROPIC_MODEL=claude-opus-5 node curriculum-squad.js "Comparative Mythology"
```

### Use OpenAI instead

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-5.6-terra  # optional

node curriculum-squad.js "Comparative Mythology"
```

Both providers support web search for the research and verification tasks.

### Search limits

Citation verification is search-heavy. Tune if needed:

```bash
MAX_SEARCHES=50 MAX_SEARCH_TURNS=60 node curriculum-squad.js "your subject"
```

## How it works

It's just functions. No classes, no framework magic.

- `agents` - Object defining each agent's name, role, goal, and background
- `tasks` - Array of tasks whose prompts are functions of `(inputs, results)`, so each can read every prior agent's output
- `callLLM()` - Handles API calls, with or without web search
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
