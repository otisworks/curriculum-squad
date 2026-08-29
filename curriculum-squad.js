#!/usr/bin/env node
/**
 * Curriculum Design Team Runner
 * Give it a subject, get back a researched, verified course proposal.
 * No framework BS - just agents, tasks, and results.
 *
 * Supports: Anthropic, OpenAI, OpenRouter, DeepSeek
 */

import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";
import { readFileSync, existsSync, realpathSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

// ============================================
// ENV FILE LOADING
// ============================================

/**
 * Minimal .env reader - deliberately not a dependency.
 *
 * Exists so someone can save their API key once in a file instead of running
 * `export` in every new terminal. That ritual is the single most common reason
 * a first run fails for someone who doesn't live in a shell.
 *
 * Real environment variables always win, so `KEY=... node curriculum-squad.js`
 * still overrides the file.
 */
function parseEnvFile(contents) {
  const vars = {};

  for (const rawLine of contents.split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;

    // Tolerate "export KEY=value", since people paste that out of instructions
    const withoutExport = line.startsWith("export ") ? line.slice(7).trim() : line;

    const eq = withoutExport.indexOf("=");
    if (eq === -1) continue;

    const key = withoutExport.slice(0, eq).trim();
    if (!key) continue;

    let value = withoutExport.slice(eq + 1).trim();

    // Strip matching surrounding quotes; leave inner content alone
    const quoted =
      (value.startsWith('"') && value.endsWith('"') && value.length > 1) ||
      (value.startsWith("'") && value.endsWith("'") && value.length > 1);
    if (quoted) {
      value = value.slice(1, -1);
    } else {
      // Only strip trailing comments on unquoted values - a key could contain '#'
      const hash = value.indexOf(" #");
      if (hash !== -1) value = value.slice(0, hash).trim();
    }

    vars[key] = value;
  }

  return vars;
}

function loadEnvFile() {
  const scriptDir = dirname(fileURLToPath(import.meta.url));

  // Later entries win, so a project-local .env beats one next to the script
  const candidates = [join(scriptDir, ".env"), join(process.cwd(), ".env")];
  const loaded = [];

  for (const path of candidates) {
    if (!existsSync(path)) continue;
    try {
      const vars = parseEnvFile(readFileSync(path, "utf8"));
      for (const [key, value] of Object.entries(vars)) {
        // Never clobber a real environment variable
        if (process.env[key] === undefined) process.env[key] = value;
      }
      loaded.push(path);
    } catch {
      // A malformed .env should never be the reason the tool won't start
    }
  }

  return loaded;
}

const LOADED_ENV_FILES = loadEnvFile();

// ============================================
// CONFIGURATION
// ============================================

/**
 * Provider registry.
 *
 * `sdk` selects the client. "anthropic" uses the Anthropic SDK; "openai" uses the
 * OpenAI SDK, which also drives OpenRouter and DeepSeek since both expose an
 * OpenAI-compatible endpoint. Only the baseURL differs.
 *
 * `search` is the field that actually matters here:
 *   "native" - server-side agentic search; the model searches repeatedly until satisfied.
 *   "plugin" - OpenRouter's `web` plugin. Grounds each request with a fixed set of
 *              results: one pass per request, not an agentic loop. Real search, but
 *              thinner than native when checking many citations one at a time.
 *   "none"   - no web access. Research quality drops and verification becomes
 *              unsound; see the no-search prompt variants below.
 */
const PROVIDERS = {
  anthropic: {
    label: "Anthropic",
    sdk: "anthropic",
    apiKeyEnv: "ANTHROPIC_API_KEY",
    modelEnv: "ANTHROPIC_MODEL",
    defaultModel: "claude-sonnet-5",
    search: "native",
  },
  openai: {
    label: "OpenAI",
    sdk: "openai",
    apiKeyEnv: "OPENAI_API_KEY",
    modelEnv: "OPENAI_MODEL",
    defaultModel: "gpt-5.6-terra",
    search: "native",
  },
  openrouter: {
    label: "OpenRouter",
    sdk: "openai",
    baseURL: "https://openrouter.ai/api/v1",
    apiKeyEnv: "OPENROUTER_API_KEY",
    modelEnv: "OPENROUTER_MODEL",
    defaultModel: "anthropic/claude-sonnet-5",
    search: "plugin",
  },
  deepseek: {
    label: "DeepSeek",
    sdk: "openai",
    baseURL: "https://api.deepseek.com",
    apiKeyEnv: "DEEPSEEK_API_KEY",
    modelEnv: "DEEPSEEK_MODEL",
    defaultModel: "deepseek-v4-pro",
    search: "none",
  },
};

const CONFIG = {
  provider: process.env.LLM_PROVIDER || "anthropic",
  maxTokens: 16384,
  // Citation verification burns through searches faster than essay fact-checking did
  maxSearches: Number(process.env.MAX_SEARCHES) || 30,
  // Hard stop on the agentic loop so a misbehaving model can't bill indefinitely
  maxSearchTurns: Number(process.env.MAX_SEARCH_TURNS) || 40,
  // Results per grounding pass for OpenRouter's web plugin
  webPluginResults: Number(process.env.WEB_PLUGIN_RESULTS) || 5,
  // Below this, an agent's output is almost certainly truncated or refused
  minOutputChars: Number(process.env.MIN_OUTPUT_CHARS) || 2000,
};

/**
 * Resolve the active provider: definition, model, and key.
 * Fails with an actionable message instead of letting a vague 401 surface later.
 */
function resolveProvider(name = CONFIG.provider) {
  const provider = PROVIDERS[name];
  if (!provider) {
    throw new Error(
      `Unknown provider "${name}". Options: ${Object.keys(PROVIDERS).join(", ")}`
    );
  }

  const apiKey = process.env[provider.apiKeyEnv];
  if (!apiKey) {
    throw new Error(
      `Missing API key for ${provider.label}.\n  export ${provider.apiKeyEnv}="..."`
    );
  }

  return {
    name,
    ...provider,
    apiKey,
    model: process.env[provider.modelEnv] || provider.defaultModel,
  };
}

function providerSupportsSearch(name = CONFIG.provider) {
  return (PROVIDERS[name]?.search ?? "none") !== "none";
}

const SEARCH_LABELS = {
  native: "native (agentic, model searches until satisfied)",
  plugin: "web plugin (one grounding pass per request)",
  none: "NONE - citations cannot be verified",
};

/**
 * Shown at startup on a provider without web access. Deliberately blunt: the
 * failure it describes is invisible in the output, which is what makes it
 * dangerous. The proposal gets a matching banner written into the file itself.
 */
const SEARCH_UNAVAILABLE_NOTICE = `
${"!".repeat(60)}
WARNING: ${"this provider has no web search"}

Archive will research from training data only - possibly stale, and it
cannot confirm any source exists.

Critic CANNOT VERIFY CITATIONS on this run. It has been instructed not to
use the word VERIFIED at all, because checking a citation against the same
model that produced it proves nothing.

Treat every source in the output as unconfirmed until a human checks it.
For a run you intend to act on, use anthropic, openai, or openrouter.
${"!".repeat(60)}`;

// ============================================
// AGENTS
// ============================================

const agents = {
  archive: {
    name: "Archive",
    role: "Subject Research Librarian",
    goal: "Map the intellectual terrain of a subject: its canon, its critics, its live debates, its practitioners, and how it is actually taught right now.",
    background: `Subject-liaison librarian crossed with a historian of ideas.
Knows how to find real syllabi in the wild, trace how a field's canon was assembled and who assembled it,
and tell a foundational text from a merely fashionable one.
Cites precisely - author, title, publisher, year - because a citation that cannot be checked is worthless.
Pays close attention to what a field habitually leaves out.`,
  },

  dean: {
    name: "Dean",
    role: "Curriculum Architect",
    goal: "Turn raw research into a teachable course with a defensible intellectual arc, measurable outcomes, and pacing that survives contact with a real academic calendar.",
    background: `Veteran course designer and department chair. Fluent in backward design, Bloom's taxonomy,
scaffolding, and cognitive load. Has watched enough courses fail to know that ambition is cheap and pacing is hard.
Believes the difference between a topic list and a curriculum is that a curriculum makes an argument:
each week earns the next one.`,
  },

  praxis: {
    name: "Praxis",
    role: "Assignment & Materials Designer",
    goal: "Build the working apparatus of the course - weekly readings, assignments, and rubrics that genuinely measure the stated learning outcomes.",
    background: `Instructional designer who has spent real time in a classroom.
Designs assignments students learn from rather than merely complete, and sequences low-stakes practice
into high-stakes deliverables so nobody hits a final project cold.
Counts pages and estimates hours honestly. Treats cost and access to materials as a design constraint, not an afterthought.`,
  },

  critic: {
    name: "Critic",
    role: "Source Verifier & Curriculum Reviewer",
    goal: "Verify every citation and pressure-test the course for workload realism, outcome alignment, and coverage before it reaches a curriculum committee.",
    background: `Sits on a curriculum review committee and has a librarian's nose for a fabricated citation.
Checks that the book exists, that the named author wrote it, that the year and publisher are right.
Equally alert to the softer failure modes: outcomes nothing actually assesses, a reading load no human could carry,
week 9 depending on a skill the course never taught. Blunt, specific, and constructive.`,
  },

  palette: {
    name: "Palette",
    role: "Visual & Media Curator",
    goal: "Identify the images, diagrams, screenings, and physical artifacts that make the course's abstract concepts concrete and memorable.",
    background: `Visual thinker and media librarian. Understands that images teach -
that one well-built diagram can collapse a week of confusion, and that students remember the artifact
long after they have forgotten the lecture.
Thinks in slide decks, screenings, object-based learning, and concept maps.
Knows the difference between an image you can use and an image you merely found, and takes rights and alt-text seriously.`,
  },
};

// ============================================
// TASKS
// ============================================

/**
 * Every agent needs the course parameters, so build the brief once.
 */
// Run context passed to every prompt builder. Defaults to the optimistic case so
// tasks can still be built standalone (tests, tooling) without a live provider.
const DEFAULT_CTX = { searchAvailable: true };

const courseBrief = (inputs) => `COURSE BRIEF
Subject: ${inputs.subject}
Level: ${inputs.level}
Length: ${inputs.weeks} weeks
Format: ${inputs.format}`;

/**
 * Search-availability preamble for Archive.
 * Without search it can still produce a useful survey from training data, but it
 * must not present that as current or confirmed.
 */
const researchSearchNotice = (searchAvailable) =>
  searchAvailable
    ? `YOU HAVE ACCESS TO WEB SEARCH. Use it aggressively. Search repeatedly with different queries.
Look specifically for real course syllabi, university catalog listings, review articles,
recent scholarship, and professional or accreditation standards where they apply.`
    : `YOU HAVE NO WEB ACCESS ON THIS RUN. Everything you write comes from training data,
which means it may be outdated and you cannot confirm that any source exists as you describe it.

Work within that limit honestly:
- Prefer works you are highly confident about - widely taught, frequently cited, long in print.
- Mark every citation [UNVERIFIED]. You have no way to check any of them.
- Say plainly when your knowledge of recent work is likely to be stale.
- Do not manufacture specificity. A half-remembered page number or ISBN is worse than none.
- Where you would normally search, note what an instructor should look up themselves.`;

/**
 * Search-availability preamble for Critic.
 *
 * This is the load-bearing one. Verification without search is not verification -
 * the model would be checking citations against the same memory that produced
 * them. So the status vocabulary changes: VERIFIED is removed entirely, because
 * nothing on this run can earn it.
 */
const verifySearchNotice = (searchAvailable) =>
  searchAvailable
    ? `YOU HAVE ACCESS TO WEB SEARCH. Use it heavily. Your primary job is catching citations that
do not exist, and you cannot do that from memory.`
    : `YOU HAVE NO WEB ACCESS ON THIS RUN, which places a hard limit on what you can honestly claim.

You cannot confirm that any cited work exists. Checking a citation against your own memory is
not verification - that memory is what generated the citation in the first place, so agreement
proves nothing. A confident VERIFIED here would be actively harmful: it would launder an
unchecked reading list into one that looks audited.

Therefore, on this run:
- The label VERIFIED is unavailable to you. Do not use it under any circumstance.
- Use only: PLAUSIBLE (consistent with what you know, still unchecked),
  SUSPECT (something is off - wrong era, wrong author, oddly specific, too convenient),
  or UNCHECKABLE (you have no useful knowledge of this work).
- Open your report by stating that no citation in it has been externally verified.
- Rank the citations by how urgently a human should check them, riskiest first.

Everything that does not depend on external lookup - outcome alignment, workload arithmetic,
internal sequencing, rubric clarity, coverage - you can still assess rigorously. Do that fully.`;

/**
 * Repeated in every prompt that touches a citation. Fabricated sources are
 * the single most damaging failure mode for a syllabus.
 */
const CITATION_RULE = `CITATION DISCIPLINE (non-negotiable):
Give full citations - author, title, publisher, year, and DOI or ISBN when you know it.
Never invent a source. If you are not certain a work exists exactly as you are describing it,
mark it [UNVERIFIED] and say what you are unsure about. An honest gap is useful; a plausible
fabrication wastes a reviewer's afternoon and can sink the whole proposal.`;

const tasks = [
  {
    id: "research",
    agent: "archive",
    useWebSearch: true, // Enable web search for this task
    description: (inputs, results, ctx = DEFAULT_CTX) => `${courseBrief(inputs)}

Survey the field so a course can be built on top of your findings.

${researchSearchNotice(ctx.searchAvailable)}

${CITATION_RULE}

Produce a structured research dossier with these sections:

1. FIELD DEFINITION & SCOPE
   What the subject actually covers, where its boundaries are contested, and which adjacent
   fields it borrows from or feeds into.

2. INTELLECTUAL HISTORY
   How the field developed, its major turning points, and the arguments that reorganized it.

3. KEY THINKERS & PRACTITIONERS
   Names, dates, affiliations, and one line on why each matters to a student.
   Include living practitioners, not only dead theorists.

4. FOUNDATIONAL TEXTS
   The works a course in this subject is expected to engage. Full citations.
   Note roughly how long and how difficult each is.

5. CONTEMPORARY & EMERGING WORK
   Significant scholarship, practice, or tooling from the last 5-10 years.
   This is what keeps a course from feeling embalmed.

6. LIVE DEBATES & COMPETING SCHOOLS
   What practitioners currently disagree about, and what is at stake in the disagreement.
   These are where the best seminar discussions come from.

7. HOW IT IS CURRENTLY TAUGHT
   Real courses at real institutions where you can find them. Typical structures, common
   sequencing, usual prerequisites, and standard assessment patterns.

8. GAPS, BLIND SPOTS & UNDERREPRESENTED VOICES
   Who the standard canon omits and what a more complete version would include.
   Be specific and name alternatives rather than gesturing at the problem.

9. PRACTICAL & APPLIED DIMENSIONS
   Methods, tools, software, fieldwork, labs, or studio practice a student should touch.

10. SOURCE LIST
    Everything you cited above, consolidated, with a confidence note on each.

Depth over breadth. A curriculum built on a shallow survey will be a shallow course.`,
    expectedOutput: `Structured research dossier covering field scope, intellectual history, key figures,
foundational and contemporary texts, live debates, current teaching practice, canon gaps,
applied dimensions, and a consolidated source list with confidence notes.`,
  },

  {
    id: "outline",
    agent: "dean",
    description: (inputs, results) => `${courseBrief(inputs)}

Here is the research dossier for the subject:

${results.research}

Design the course.

Not a list of topics - a curriculum that makes an argument, where each week earns the next.
Decide what this course is FOR, then build backward from that.

Produce the following, in order:

## COURSE TITLE
A real title plus a short subtitle if it helps.

## CATALOG DESCRIPTION
Roughly 150 words, written the way it would appear in the course catalog.

## COURSE RATIONALE
Why this course exists, why it is worth ${inputs.weeks} weeks of a student's life, and what
argument the sequence of units makes. Be explicit about the through-line.

## TARGET AUDIENCE & PREREQUISITES
Who this is for at the ${inputs.level} level, what they need coming in, and what you are
deliberately NOT assuming.

## LEARNING OUTCOMES
4-6 course-level outcomes. Use measurable verbs - analyze, construct, evaluate, critique.
Avoid "understand" and "appreciate"; nobody can assess those.
Number them LO1, LO2, etc. so later work can reference them.

## STRUCTURAL OVERVIEW
How the ${inputs.weeks} weeks divide into units or arcs, and the logic of that division.
State the pivot points where the course changes gear.

## WEEK-BY-WEEK SCHEDULE
All ${inputs.weeks} weeks. Do not skip or abbreviate the back half - late weeks are where
courses usually fall apart. For each week give:
- Week number and title
- The guiding question the week answers
- Topics covered
- 2-4 session-level objectives
- Which course outcome(s) it advances
- How it builds on the previous week and sets up the next

## ASSESSMENT STRATEGY
The shape of assessment and the reasoning behind it, plus a grading breakdown with
percentage weights totaling 100. Leave the detailed assignment design to the next agent -
you are setting the frame.

## COURSE POLICIES
Brief notes on participation, late work, generative AI use, and accessibility.

## INSTRUCTOR PREP NOTES
Where this course is genuinely hard to teach: the weeks students reliably struggle with,
the material that needs the most scaffolding, and anything requiring lead time to arrange.

Respect the format. A ${inputs.format} paces very differently from a lecture course -
build for how this room actually runs.`,
    expectedOutput: `Complete course outline: title, catalog description, rationale, prerequisites,
numbered measurable learning outcomes, unit structure, full week-by-week schedule with guiding
questions and objectives, assessment strategy with grading weights, policies, and instructor prep notes.`,
  },

  {
    id: "materials",
    agent: "praxis",
    description: (inputs, results) => `${courseBrief(inputs)}

Research dossier:

${results.research}

Course outline to build against:

${results.outline}

Build the working apparatus of this course: what students read, what they make, and how it is graded.

${CITATION_RULE}

Produce the following:

## A. WEEKLY READING LIST
Every week, all ${inputs.weeks} of them. For each week:
- Required readings with full citations and page ranges
- Estimated reading time, given a ${inputs.level} student
- Optional / "go deeper" readings
- Access note: open access, library-licensed, in print, out of print, or paywalled
Keep the load consistent and defensible. If one week spikes, say so and justify it.

## B. MEDIA & PRIMARY SOURCES
Per week where relevant: films, recordings, datasets, archives, artifacts, documentation,
software. Include where to actually get them.

## C. MAJOR ASSIGNMENTS
3-5 of them. For each:
- Title
- The prompt exactly as a student would receive it
- Which learning outcomes it measures, by number
- Deliverable format and length
- Which week it is assigned and which week it is due
- Milestones or checkpoints along the way
- Percentage of final grade
The weights here must reconcile with the grading breakdown in the outline. If they do not,
flag the discrepancy explicitly rather than silently changing it.

## D. RUBRICS
A rubric for each major assignment. Criteria down the side, performance levels across the top,
with descriptions concrete enough that two graders would land in the same place.

## E. WEEKLY LOW-STAKES WORK
The recurring small stuff: response papers, problem sets, studio exercises, discussion leads,
annotation. Explain how it scaffolds toward the major assignments.

## F. FINAL PROJECT SPECIFICATION
Full spec with a milestone schedule mapped to specific weeks, and a description of what an
excellent one looks like versus a merely adequate one.

## G. WORKLOAD AUDIT
Estimated student hours per week: reading, assignments, contact time, total.
Compare against what is reasonable for ${inputs.level}. Be honest if it is too much.

## H. ALTERNATE PATHWAYS
2-3 substitute assignments for different modalities, accessibility needs, or students
arriving with different backgrounds. Equivalent rigor, different route.

Design assignments students learn from, not assignments students merely complete.`,
    expectedOutput: `Complete course materials: week-by-week reading list with access notes and time
estimates, media and primary sources, 3-5 fully specified major assignments with rubrics,
low-stakes weekly work, final project spec with milestones, workload audit, and alternate pathways.`,
  },

  {
    id: "verify",
    agent: "critic",
    useWebSearch: true, // Enable web search for citation verification
    description: (inputs, results, ctx = DEFAULT_CTX) => `${courseBrief(inputs)}

${verifySearchNotice(ctx.searchAvailable)}

Course outline:

${results.outline}

Course materials:

${results.materials}

Underlying research:

${results.research}

Review this course the way a curriculum committee would, but do the citation work a committee
never has time for.

## 1. CITATION VERIFICATION
Go through EVERY cited work in the outline and materials.
Present results as a table with columns: Cited As | Status | Correction / Notes

${
  ctx.searchAvailable
    ? `Search for each one. Use exactly these status values:
- VERIFIED - exists as cited
- CORRECTED - exists, but a detail was wrong (give the corrected citation in full)
- UNVERIFIABLE - could not confirm either way (say what you searched)
- LIKELY FABRICATED - strong evidence this work does not exist`
    : `You have no search access, so use exactly these status values:
- PLAUSIBLE - consistent with what you know, but UNCHECKED
- SUSPECT - something is wrong or suspiciously convenient about this citation
- UNCHECKABLE - you have no useful knowledge of this work
Do not use VERIFIED. Nothing on this run has been verified.`
}

Do not skim this section. A single invented book in a reading list discredits the entire proposal.
List every fabrication and every correction again in a consolidated block at the end of this
section so they are easy to act on.

## 2. ACCESS & COST AUDIT
Which readings are genuinely obtainable? Flag anything out of print, prohibitively expensive,
or paywalled without a clear institutional route. Suggest open-access substitutes where they exist.
Estimate total out-of-pocket cost to a student.

## 3. OUTCOME ALIGNMENT
Build a matrix of learning outcomes against assessments.
Identify orphan outcomes (stated but never assessed) and orphan assessments (graded work that
measures no stated outcome). Both are committee red flags.

## 4. WORKLOAD REALISM
Check the estimated hours against the actual assigned pages and deliverables.
Reading time estimates in course proposals are routinely optimistic - recalculate them yourself
and say plainly whether this is survivable at ${inputs.level}.

## 5. SEQUENCING & PREREQUISITES
Does every week have what it needs from earlier weeks? Find any place where the course
depends on a concept, skill, or tool it has not yet taught. Check that assignment due dates
land after the material required to do them.

## 6. COVERAGE & PERSPECTIVE
What is missing that a specialist would immediately notice? Is the reading list narrow in
authorship, geography, method, or period? Name specific works that would fix it.

## 7. ASSESSMENT VALIDITY
Are the rubrics specific enough to grade consistently? Do the grading weights sum to 100 and
match between outline and materials? Is anything unusually vulnerable to generative-AI shortcuts,
and can it be redesigned rather than policed?

## 8. VERDICT
One of: READY TO SUBMIT / REVISE BEFORE SUBMISSION / MAJOR ISSUES.
Then the prioritized list of fixes - the must-fix items first, clearly separated from the
nice-to-haves. Be direct. Vague praise here helps nobody.`,
    expectedOutput: `Verification report with a full citation status table and consolidated corrections,
access and cost audit, outcome-to-assessment alignment matrix, recalculated workload check,
sequencing audit, coverage critique, assessment validity review, and a prioritized verdict.`,
  },

  {
    id: "visuals",
    agent: "palette",
    description: (inputs, results) => `${courseBrief(inputs)}

Course outline:

${results.outline}

Course materials:

${results.materials}

Build the visual and media layer of this course. Not decoration - teaching instruments.
For every recommendation, the test is: what does this let a student see that prose does not?

Produce the following:

## COURSE VISUAL IDENTITY
The overall look for slides, handouts, and the LMS space: palette, typography mood,
diagram style, and the general register. A ${inputs.format} on this subject should feel
like something specific. Say what.

## WEEK-BY-WEEK VISUAL PLAN
For each week of the course:
- The key visual or visuals
- What it depicts
- What it teaches that text alone would not
- Where it lands in the session - cold open, mid-lecture, discussion prompt, closing
- Medium: archival photograph, diagram, map, data visualization, film clip, artifact, screenshot
- 2-3 specific real reference examples, named
- Where to source it: which archive, museum, collection, or database
- Rights and licensing note

## THE COURSE SPINE
One anchoring image per week that, laid out in sequence, tells the story of the whole course
at a glance. This doubles as a review tool and as the poster for the course.

## RECURRING DIAGRAMS
The structural visuals worth building once and returning to all term: concept maps, timelines,
comparison matrices, process diagrams, family trees of influence. Describe each well enough
that someone could actually draw it.

## SCREENINGS & MEDIA
Films, documentaries, recordings, lectures, or interactive works worth assigning or showing.
Include runtime and where to legally access each.

## OBJECT-BASED & EXPERIENTIAL
Physical artifacts, site visits, lab or studio demonstrations, guest practitioners, and
collections worth building a session around. Note lead time required to arrange them.

## ACCESSIBILITY
Alt-text approach for the core images, captioning needs for media, and how each visual
argument reaches a student who cannot see it. Every visual above needs a non-visual path
to the same idea.

Be specific. "A photo of the era" is useless; name the photograph, the photographer, and
the collection that holds it. If you cannot name a real example, describe precisely what to
search for and mark it [NEEDS SOURCING] rather than inventing a plausible-sounding artifact.`,
    expectedOutput: `Visual and media plan: course visual identity, week-by-week visual recommendations
with named references and sourcing, a one-image-per-week course spine, recurring structural diagrams,
screening list, object-based learning opportunities, and accessibility notes.`,
    optional: "visuals", // Can be skipped with --no-visuals
  },
];

// ============================================
// LLM CLIENT
// ============================================

function createClient(provider = resolveProvider()) {
  if (provider.sdk === "anthropic") {
    return new Anthropic({ apiKey: provider.apiKey });
  }

  // OpenAI, OpenRouter, and DeepSeek all speak the OpenAI protocol
  return new OpenAI({
    apiKey: provider.apiKey,
    ...(provider.baseURL ? { baseURL: provider.baseURL } : {}),
    ...(provider.name === "openrouter"
      ? {
          defaultHeaders: {
            "HTTP-Referer": "https://github.com/otisworks/curriculum-squad",
            "X-Title": "curriculum-squad",
          },
        }
      : {}),
  });
}

/**
 * A response that stopped early is not a result, it's a fragment - and a fragment
 * of a reading list is worse than none, because every downstream agent treats it
 * as complete. Detect it, mark it retryable, and describe it in plain language.
 */
function assertComplete(agent, text, stopReason) {
  const truncated =
    stopReason === "max_tokens" ||
    stopReason === "length" ||
    stopReason === "incomplete";

  if (truncated) {
    const error = new Error(
      `${agent.name} ran out of room and its answer was cut off mid-sentence.`
    );
    error.retryable = true;
    error.guidance =
      `This usually means the course is too big to describe in one response.\n` +
      `Try asking for a shorter course, for example --weeks=10.`;
    throw error;
  }

  const length = text ? text.trim().length : 0;
  if (length < CONFIG.minOutputChars) {
    const error = new Error(
      `${agent.name} stopped early, returning only ${length} characters where a ` +
        `complete answer runs to several thousand.`
    );
    error.retryable = true;
    error.guidance =
      `This is usually a temporary hiccup from the AI provider rather than a\n` +
      `problem with your subject. Running the same command again often works.`;
    throw error;
  }

  return text;
}

// ============================================
// ERROR TRANSLATION
// ============================================

/**
 * Turn provider errors into something a curriculum designer can act on.
 *
 * The people using this are not going to read a raw 401 JSON body and conclude
 * that their API key has a stray space in it. Each case returns what happened
 * and what to do about it; --debug still shows the original.
 */
function explainError(error, provider) {
  const status = error?.status ?? error?.response?.status;
  const raw = `${error?.message ?? error}`;
  const keyEnv = provider?.apiKeyEnv ?? "your API key";
  const modelEnv = provider?.modelEnv ?? "the model variable";
  const label = provider?.label ?? "the AI provider";

  // Our own errors already speak human and carry their own guidance
  if (error?.guidance) {
    return { headline: raw, guidance: error.guidance, retryable: !!error.retryable };
  }

  if (status === 401 || status === 403 || /authentication|invalid.*api.?key|unauthorized/i.test(raw)) {
    return {
      headline: `${label} rejected your API key.`,
      guidance:
        `Check that ${keyEnv} is set correctly.\n` +
        `Common causes: a missing character from copy/paste, an extra space,\n` +
        `a key for a different provider, or a key that has been revoked.\n\n` +
        `You can put it in a file named .env next to this script:\n` +
        `  ${keyEnv}=your-key-here`,
      retryable: false,
    };
  }

  if (status === 404 || /model.*(not found|does not exist)|unknown model/i.test(raw)) {
    return {
      headline: `${label} doesn't recognise the model "${provider?.model}".`,
      guidance:
        `The model name may have changed or may not be available on your account.\n` +
        `Set a different one with ${modelEnv}, or unset it to use the default.`,
      retryable: false,
    };
  }

  if (status === 429 || /rate.?limit|too many requests/i.test(raw)) {
    return {
      headline: `${label} is rate limiting this account.`,
      guidance:
        `You've sent requests faster than your plan allows, or hit a usage cap.\n` +
        `Waiting a few minutes and running the same command again usually works.`,
      retryable: true,
    };
  }

  if (/insufficient|quota|billing|credit|payment/i.test(raw)) {
    return {
      headline: `${label} reports a billing or credit problem.`,
      guidance:
        `The account behind this API key may be out of credit.\n` +
        `Check the billing page for ${label} and top up if needed.`,
      retryable: false,
    };
  }

  if (/context length|too long|maximum.*tokens|prompt is too long/i.test(raw)) {
    return {
      headline: `The request grew too large for ${label} to handle.`,
      guidance:
        `Earlier steps produced more material than the model can read back in.\n` +
        `Try a shorter course, for example --weeks=10.`,
      retryable: false,
    };
  }

  if (status >= 500 || /overloaded|server error|service unavailable|timeout|ETIMEDOUT/i.test(raw)) {
    return {
      headline: `${label} is having trouble on their end.`,
      guidance:
        `This is a temporary problem with the provider, not with your setup.\n` +
        `Waiting a few minutes and running the same command again usually works.`,
      retryable: true,
    };
  }

  if (/ENOTFOUND|ECONNREFUSED|ECONNRESET|fetch failed|network/i.test(raw)) {
    return {
      headline: `Couldn't reach ${label}.`,
      guidance:
        `This looks like a network problem. Check your internet connection.\n` +
        `If you're on a work network, a firewall may be blocking the connection -\n` +
        `that's one your IT team can help with.`,
      retryable: true,
    };
  }

  return {
    headline: `Something went wrong while talking to ${label}.`,
    guidance:
      `The error was: ${raw}\n\n` +
      `Run the same command with --debug to see the full technical details.`,
    retryable: false,
  };
}

/**
 * Print a failure the way a non-technical user needs to see it.
 */
function reportError(error, provider, debug = false) {
  const { headline, guidance } = explainError(error, provider);

  console.error(`\n${"-".repeat(60)}`);
  console.error(`STOPPED: ${headline}`);
  console.error(`${"-".repeat(60)}\n`);
  if (guidance) console.error(`${guidance}\n`);

  if (debug) {
    console.error(`${"-".repeat(60)}`);
    console.error("Technical detail (--debug):");
    console.error(error?.stack ?? error);
    console.error(`${"-".repeat(60)}\n`);
  }
}

/**
 * Run a step, retrying once if the failure looks transient.
 *
 * A run is many minutes of API time and real money. Dying on a hiccup that a
 * single retry would have absorbed is the worst possible outcome for someone
 * who has been watching a terminal for a quarter of an hour.
 */
async function withRetry(fn, { provider, verbose = true, attempts = 2, delayMs = 3000 } = {}) {
  let lastError;

  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      return await fn(attempt);
    } catch (error) {
      lastError = error;
      const { headline, retryable } = explainError(error, provider);

      if (!retryable || attempt === attempts) break;

      if (verbose) {
        console.log(`\n  ${headline}`);
        console.log(`  Trying that step once more in ${delayMs / 1000}s...\n`);
      }
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }

  throw lastError;
}

/**
 * Build an agent's system prompt. Single source of truth for all call paths.
 */
function buildSystemPrompt(agent, withSearch = false) {
  const base = `You are ${agent.name}, a ${agent.role}.

Your goal: ${agent.goal}

Background: ${agent.background}

Stay in character. Be thorough and specific.`;

  if (!withSearch) return base;

  return `${base}

You have access to web search. Use it to find information, verify facts, and discover obscure sources.
Search multiple times with different queries to get comprehensive results.`;
}

/**
 * Call LLM without tools (simple completion)
 */
async function callLLMSimple(client, agent, prompt, provider = resolveProvider()) {
  const systemPrompt = buildSystemPrompt(agent);

  if (provider.sdk === "anthropic") {
    const response = await client.messages.create({
      model: provider.model,
      max_tokens: CONFIG.maxTokens,
      system: systemPrompt,
      messages: [{ role: "user", content: prompt }],
    });
    // Extract text from response, handling different content block types
    const textBlocks = response.content.filter((block) => block.type === "text");
    if (textBlocks.length === 0) {
      throw new Error(`No text in response. Got: ${JSON.stringify(response.content.map(b => b.type))}`);
    }
    return assertComplete(
      agent,
      textBlocks.map((b) => b.text).join("\n"),
      response.stop_reason
    );
  } else {
    const response = await client.chat.completions.create({
      model: provider.model,
      max_completion_tokens: CONFIG.maxTokens,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: prompt },
      ],
    });
    return assertComplete(
      agent,
      response.choices[0].message.content,
      response.choices[0].finish_reason
    );
  }
}

/**
 * Call Anthropic with web search capability
 * Uses an agentic loop to handle tool calls
 */
async function callAnthropicWithSearch(client, agent, prompt, verbose = true, provider = resolveProvider()) {
  const systemPrompt = buildSystemPrompt(agent, true);

  // Anthropic's server-side web search tool
  const tools = [
    {
      type: "web_search_20250305",
      name: "web_search",
      // Citation verification needs a lot of individual lookups
      max_uses: CONFIG.maxSearches,
    },
  ];

  let messages = [{ role: "user", content: prompt }];
  let searchCount = 0;
  let turns = 0;

  // Agentic loop - keep going until model stops using tools.
  // Capped: a model that never emits end_turn would otherwise bill forever.
  while (turns < CONFIG.maxSearchTurns) {
    turns++;
    const response = await client.messages.create({
      model: provider.model,
      max_tokens: CONFIG.maxTokens,
      system: systemPrompt,
      tools: tools,
      messages: messages,
    });

    // Check if we're done (no more tool use)
    if (response.stop_reason === "end_turn") {
      // Extract final text response
      const textBlocks = response.content.filter(
        (block) => block.type === "text"
      );
      return assertComplete(
        agent,
        textBlocks.map((b) => b.text).join("\n"),
        response.stop_reason
      );
    }

    // Process tool uses
    const toolUseBlocks = response.content.filter(
      (block) => block.type === "tool_use"
    );

    if (toolUseBlocks.length === 0) {
      // No tool use and not end_turn - extract whatever text we have
      const textBlocks = response.content.filter(
        (block) => block.type === "text"
      );
      return assertComplete(
        agent,
        textBlocks.map((b) => b.text).join("\n"),
        response.stop_reason
      );
    }

    // Add assistant's response to messages
    messages.push({ role: "assistant", content: response.content });

    // Process each tool use - for web_search, Anthropic handles it server-side
    // We just need to continue the conversation
    for (const toolUse of toolUseBlocks) {
      if (toolUse.name === "web_search") {
        searchCount++;
        if (verbose) {
          console.log(`  [Web Search #${searchCount}]: ${toolUse.input.query}`);
        }
        // Anthropic's web_search is server-side, results come back automatically
        // We don't need to provide tool_result for server-side tools
      }
    }

    // For server-side tools like web_search, we don't add tool_result messages
    // The API handles the search and includes results in the next response
    // Just continue the loop - the next API call will have search results
  }

  throw new Error(
    `Agent ${agent.name} exceeded ${CONFIG.maxSearchTurns} search turns without finishing. ` +
      `Raise CONFIG.maxSearchTurns if this is a legitimately large verification job.`
  );
}

/**
 * Call OpenAI with web search capability
 * Uses the responses API with web_search tool
 */
async function callOpenAIWithSearch(client, agent, prompt, verbose = true, provider = resolveProvider()) {
  const systemPrompt = buildSystemPrompt(agent, true);

  const fullPrompt = `${systemPrompt}\n\n${prompt}`;

  if (verbose) {
    console.log(`  [Web Search enabled via OpenAI]`);
  }

  const response = await client.responses.create({
    model: provider.model,
    tools: [{ type: "web_search" }],
    input: fullPrompt,
  });

  return assertComplete(agent, response.output_text, response.status);
}

/**
 * Call OpenRouter with its `web` plugin.
 *
 * Unlike the native paths, this is not an agentic loop: OpenRouter runs a search
 * and injects the results into the prompt before the model responds. The model
 * cannot decide to search again after reading them. That's adequate for Archive's
 * broad survey, but genuinely thinner for Critic, which ideally looks up each
 * citation separately. Documented in the README rather than papered over.
 */
async function callOpenRouterWithSearch(client, agent, prompt, verbose = true, provider = resolveProvider()) {
  const systemPrompt = buildSystemPrompt(agent, true);

  if (verbose) {
    console.log(
      `  [Web plugin enabled via OpenRouter, max_results=${CONFIG.webPluginResults}]`
    );
  }

  const response = await client.chat.completions.create({
    model: provider.model,
    max_completion_tokens: CONFIG.maxTokens,
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: prompt },
    ],
    // OpenRouter-specific: model-agnostic grounding, native where available and
    // Exa-backed otherwise. Passed through by the OpenAI SDK untouched.
    plugins: [{ id: "web", max_results: CONFIG.webPluginResults }],
  });

  const choice = response.choices[0];

  // Surface the sources the plugin actually used, so a reader can audit them
  const annotations = choice.message?.annotations ?? [];
  const citations = annotations
    .filter((a) => a.type === "url_citation")
    .map((a) => `- ${a.url_citation.title}: ${a.url_citation.url}`);

  if (verbose && citations.length) {
    console.log(`  [${citations.length} sources returned by web plugin]`);
  }

  const text = assertComplete(agent, choice.message.content, choice.finish_reason);

  return citations.length
    ? `${text}\n\n---\n\n## Sources retrieved by web search\n\n${citations.join("\n")}`
    : text;
}

/**
 * Call LLM with web search capability
 * Routes to the appropriate implementation for the provider's search style
 */
async function callLLMWithSearch(client, agent, prompt, verbose = true, provider = resolveProvider()) {
  if (provider.search === "plugin") {
    return callOpenRouterWithSearch(client, agent, prompt, verbose, provider);
  }
  if (provider.sdk === "anthropic") {
    return callAnthropicWithSearch(client, agent, prompt, verbose, provider);
  }
  return callOpenAIWithSearch(client, agent, prompt, verbose, provider);
}

/**
 * Main LLM call function - routes to appropriate handler
 */
async function callLLM(client, agent, prompt, useWebSearch = false, verbose = true, provider = resolveProvider()) {
  // A task wanting search on a provider without it falls back to a plain call.
  // The prompt has already been swapped for its no-search variant by this point.
  if (useWebSearch && provider.search !== "none") {
    return callLLMWithSearch(client, agent, prompt, verbose, provider);
  }
  return callLLMSimple(client, agent, prompt, provider);
}

// ============================================
// CLI ARGUMENT PARSING
// ============================================

const DEFAULTS = {
  level: "upper-division undergraduate",
  weeks: 15,
  format: "seminar",
};

// Shorthands so you don't have to type "upper-division undergraduate" every time
const LEVEL_ALIASES = {
  intro: "introductory undergraduate (no prior background)",
  undergrad: "general undergraduate",
  upper: "upper-division undergraduate",
  grad: "graduate",
  phd: "doctoral",
};

const USAGE = `Usage: node curriculum-squad.js "<subject>" [options]

Options:
  --level=<level>    intro | undergrad | upper | grad | phd, or any free-text description
                     (default: ${DEFAULTS.level})
  --weeks=<n>        Course length in weeks, 1-52 (default: ${DEFAULTS.weeks})
  --format=<format>  seminar | lecture | studio | lab | workshop | online, or free text
                     (default: ${DEFAULTS.format})
  --provider=<name>  ${Object.keys(PROVIDERS).join(" | ")}
                     (default: ${CONFIG.provider}, or set LLM_PROVIDER)
  --no-visuals       Skip the Palette agent
  --debug            Show full technical detail if something fails
  -h, --help         Show this message

Tip: put quotes around a subject with spaces.
  node curriculum-squad.js "History of Cartography"

API keys: set the variable for your provider below, either with "export" or by
saving it in a file named .env next to this script, e.g.
  ANTHROPIC_API_KEY=sk-ant-...

Providers:
${Object.entries(PROVIDERS)
  .map(
    ([name, p]) =>
      `  ${name.padEnd(11)} ${p.apiKeyEnv.padEnd(19)} search: ${
        p.search === "none" ? "NONE - cannot verify citations" : p.search
      }`
  )
  .join("\n")}

  DeepSeek has no web search API. Research and verification both degrade;
  Critic is blocked from claiming anything is VERIFIED. Fine for cheap drafts,
  not for a proposal you intend to submit.

Examples:
  node curriculum-squad.js "Media Archaeology"
  node curriculum-squad.js "Mycology for Artists" --level=grad --weeks=10 --format=studio
  node curriculum-squad.js "Data Ethics" --provider=openrouter --weeks=12
  node curriculum-squad.js "Cheap Draft" --provider=deepseek --no-visuals`;

function parseArgs(args) {
  const flags = {
    noVisuals: false, // --no-visuals: skip palette
    help: false,
    debug: false,
    subject: null,
    level: DEFAULTS.level,
    weeks: DEFAULTS.weeks,
    format: DEFAULTS.format,
    provider: null, // null = fall back to LLM_PROVIDER / default
  };

  // Collected rather than overwritten: an unquoted subject arrives as several
  // arguments, and silently keeping only the last one produced the wrong course.
  const subjectWords = [];

  // Accepts both --key=value and --key value
  const readValue = (arg, i) => {
    const eq = arg.indexOf("=");
    if (eq !== -1) return { value: arg.slice(eq + 1), next: i };
    return { value: args[i + 1], next: i + 1 };
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    const key = arg.split("=")[0];

    if (arg === "--no-visuals") {
      flags.noVisuals = true;
    } else if (arg === "-h" || arg === "--help") {
      flags.help = true;
    } else if (arg === "--debug") {
      flags.debug = true;
    } else if (key === "--level") {
      const { value, next } = readValue(arg, i);
      if (!value) throw new Error("--level requires a value");
      flags.level = LEVEL_ALIASES[value] || value;
      i = next;
    } else if (key === "--format") {
      const { value, next } = readValue(arg, i);
      if (!value) throw new Error("--format requires a value");
      flags.format = value;
      i = next;
    } else if (key === "--weeks") {
      const { value, next } = readValue(arg, i);
      const weeks = Number(value);
      if (!Number.isInteger(weeks) || weeks < 1 || weeks > 52) {
        throw new Error(`--weeks must be a whole number from 1 to 52 (got "${value}")`);
      }
      flags.weeks = weeks;
      i = next;
    } else if (key === "--provider") {
      const { value, next } = readValue(arg, i);
      if (!PROVIDERS[value]) {
        throw new Error(
          `Unknown provider "${value}". Options: ${Object.keys(PROVIDERS).join(", ")}`
        );
      }
      flags.provider = value;
      i = next;
    } else if (arg.startsWith("-")) {
      throw new Error(`Unknown option: ${arg}`);
    } else {
      subjectWords.push(arg);
    }
  }

  if (subjectWords.length) {
    flags.subject = subjectWords.join(" ");
    // True when quotes were forgotten - main() echoes the subject back so a
    // misread is caught in the first second rather than after a paid run
    flags.subjectWasUnquoted = subjectWords.length > 1;
  }

  return flags;
}

// ============================================
// TEAM RUNNER
// ============================================

async function runTeam(inputs, options = {}) {
  const { verbose = true, onTaskStart, onTaskComplete, skipVisuals = false } = options;
  const provider = resolveProvider();
  const client = createClient(provider);
  const results = {};

  // Passed to every prompt builder so agents know what they can honestly claim
  const ctx = { searchAvailable: provider.search !== "none" };

  // Determine which tasks to run
  const activeTasks = tasks.filter((task) => {
    if (task.optional === "visuals" && skipVisuals) return false;
    return true;
  });

  console.log(`\n${"=".repeat(60)}`);
  console.log(`CURRICULUM DESIGN TEAM`);
  console.log(`Provider: ${provider.label} (${provider.model})`);
  console.log(`Search:   ${SEARCH_LABELS[provider.search]}`);
  console.log(`Subject: ${inputs.subject}`);
  console.log(`Level:   ${inputs.level}`);
  console.log(`Length:  ${inputs.weeks} weeks`);
  console.log(`Format:  ${inputs.format}`);
  if (skipVisuals) console.log(`Mode:    NO VISUALS (skipping palette)`);
  console.log(`${"=".repeat(60)}`);

  if (!ctx.searchAvailable) {
    console.log(SEARCH_UNAVAILABLE_NOTICE);
  }
  console.log();

  for (const [index, task] of activeTasks.entries()) {
    const agent = agents[task.agent];
    const taskPrompt = task.description(inputs, results, ctx);

    if (verbose) {
      console.log(`\n[${"=".repeat(20)}]`);
      console.log(`STEP ${index + 1}/${activeTasks.length}: ${task.id.toUpperCase()}`);
      console.log(`AGENT: ${agent.name} (${agent.role})`);
      console.log(`[${"=".repeat(20)}]\n`);
    }

    if (onTaskStart) {
      onTaskStart(task, agent);
    }

    const startTime = Date.now();
    const useSearch = task.useWebSearch || false;
    if (useSearch && verbose) {
      console.log(
        ctx.searchAvailable
          ? `Web search: ENABLED (${provider.search})`
          : `Web search: UNAVAILABLE on ${provider.label} - running degraded`
      );
    }
    const result = await withRetry(
      (attempt) => {
        if (attempt > 1 && verbose) {
          console.log(`Retry ${attempt - 1} of ${task.id}...`);
        }
        return callLLM(client, agent, taskPrompt, useSearch, verbose, provider);
      },
      { provider, verbose }
    );
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

    results[task.id] = result;

    if (verbose) {
      console.log(`Completed in ${elapsed}s`);
      console.log(`Output length: ${result.length} chars`);
    }

    if (onTaskComplete) {
      await onTaskComplete(task, agent, result, elapsed);
    }
  }

  console.log(`\n${"=".repeat(60)}`);
  console.log(`CURRICULUM COMPLETE`);
  console.log(`${"=".repeat(60)}\n`);

  return results;
}

// ============================================
// CLI ENTRY POINT
// ============================================

// Filenames derive from task ids so the two can't drift apart
const OUTPUT_FILES = {
  research: "1-research.md",
  outline: "2-outline.md",
  materials: "3-materials.md",
  verify: "4-verification.md",
  visuals: "5-visuals.md",
};

/**
 * Stitch the agent outputs into one document you can hand to a committee.
 * No LLM call - just concatenation in reading order.
 */
function buildProposal(inputs, results, ctx = DEFAULT_CTX) {
  // The warning travels with the document. A console message scrolls away; this
  // stays attached to the thing someone might actually forward to a colleague.
  const provenance = ctx.searchAvailable
    ? `> Assembled by an LLM pipeline. Review the verification report before circulating -\n> citations flagged UNVERIFIABLE or LIKELY FABRICATED still need a human check.`
    : `> **NO SOURCES IN THIS DOCUMENT HAVE BEEN VERIFIED.**\n>\n> Generated on a provider without web search, so citations were produced and\n> "checked" by the same model. Every author, title, year, and ISBN below should\n> be treated as unconfirmed until a human looks it up. Do not circulate as-is.`;

  const sections = [
    `# Course Proposal: ${inputs.subject}`,
    `**Level:** ${inputs.level}  \n**Length:** ${inputs.weeks} weeks  \n**Format:** ${inputs.format}  \n**Generated:** ${new Date().toISOString().slice(0, 10)}`,
    provenance,
    `---\n\n# Course Outline\n\n${results.outline}`,
    `---\n\n# Materials & Assessment\n\n${results.materials}`,
  ];

  if (results.visuals) {
    sections.push(`---\n\n# Visual & Media Plan\n\n${results.visuals}`);
  }

  sections.push(`---\n\n# Verification Report\n\n${results.verify}`);

  return sections.join("\n\n");
}

async function main() {
  const args = process.argv.slice(2);

  let flags;
  try {
    flags = parseArgs(args);
  } catch (error) {
    console.error(`${error.message}\n\n${USAGE}`);
    process.exit(1);
  }

  if (flags.help) {
    console.log(USAGE);
    return;
  }

  // --provider overrides LLM_PROVIDER for this run
  if (flags.provider) {
    CONFIG.provider = flags.provider;
  }

  // Fail on a bad provider or missing key now, not five minutes into a run
  let provider;
  try {
    provider = resolveProvider();
  } catch (error) {
    console.error(`\n${"-".repeat(60)}`);
    console.error(`STOPPED: ${error.message.split("\n")[0]}`);
    console.error(`${"-".repeat(60)}\n`);
    const env = PROVIDERS[CONFIG.provider]?.apiKeyEnv;
    if (env) {
      console.error(`Set it for this terminal:`);
      console.error(`  export ${env}="your-key-here"\n`);
      console.error(`Or save it permanently in a file named .env next to this script:`);
      console.error(`  ${env}=your-key-here\n`);
    } else {
      console.error(`${error.message}\n`);
    }
    process.exit(1);
  }

  if (LOADED_ENV_FILES.length) {
    console.log(`Loaded settings from ${LOADED_ENV_FILES.join(", ")}`);
  }

  const subject = flags.subject || "The cultural history of maps and mapmaking";
  if (!flags.subject) {
    console.log(`No subject given, using default: "${subject}"`);
    console.log(`Run with --help to see options.\n`);
  } else if (flags.subjectWasUnquoted) {
    // Reading it back is the cheapest possible guard against a misparse
    console.log(`Subject read as: "${subject}"`);
    console.log(`(If that's not right, put quotes around it: "${subject}")\n`);
  }

  const inputs = {
    subject,
    level: flags.level,
    weeks: flags.weeks,
    format: flags.format,
  };

  const fs = await import("fs/promises");
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outputDir = `./curriculum-${timestamp}`;
  await fs.mkdir(outputDir, { recursive: true });

  try {
    // Write each result as it lands. A run is many minutes of API time -
    // a failure at the last step shouldn't discard everything before it.
    const onTaskComplete = async (task, agent, result) => {
      const filename = OUTPUT_FILES[task.id];
      if (!filename) return;
      await fs.writeFile(`${outputDir}/${filename}`, result);
      console.log(`Saved: ${outputDir}/${filename}`);
    };

    const results = await runTeam(inputs, {
      skipVisuals: flags.noVisuals,
      onTaskComplete,
    });

    const ctx = { searchAvailable: provider.search !== "none" };

    await fs.writeFile(
      `${outputDir}/course-proposal.md`,
      buildProposal(inputs, results, ctx)
    );

    console.log(`\nOutputs saved to: ${outputDir}/`);
    console.log(`Start with: ${outputDir}/course-proposal.md`);
    console.log(`Then read:  ${outputDir}/${OUTPUT_FILES.verify} (citation checks)`);

    if (!ctx.searchAvailable) {
      console.log(
        `\nReminder: this ran on ${provider.label}, which has no web search.\n` +
          `No citation in these files has been verified against anything.`
      );
    }
  } catch (error) {
    reportError(error, provider, flags.debug);

    // Don't litter the cwd with an empty directory if we failed before any output
    const written = await fs.readdir(outputDir).catch(() => []);
    if (written.length === 0) {
      await fs.rmdir(outputDir).catch(() => {});
      console.error(`Nothing was saved, so nothing was charged for beyond this attempt.\n`);
    } else {
      console.error(`The steps that finished were still saved here:`);
      console.error(`  ${outputDir}/`);
      console.error(`  (${written.sort().join(", ")})\n`);
    }
    process.exit(1);
  }
}

// Run only when executed directly, so importing this file doesn't kick off a run
const isDirectRun =
  process.argv[1] &&
  realpathSync(process.argv[1]) === realpathSync(fileURLToPath(import.meta.url));

if (isDirectRun) {
  main();
}

// Export for use as module
export {
  agents,
  tasks,
  runTeam,
  parseArgs,
  buildProposal,
  resolveProvider,
  providerSupportsSearch,
  assertComplete,
  explainError,
  withRetry,
  parseEnvFile,
  CONFIG,
  PROVIDERS,
};
