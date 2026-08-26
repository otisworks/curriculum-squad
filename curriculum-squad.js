#!/usr/bin/env node
/**
 * Curriculum Design Team Runner
 * Give it a subject, get back a researched, verified course proposal.
 * No framework BS - just agents, tasks, and results.
 *
 * Supports: Anthropic (default) or OpenAI
 */

import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";

// ============================================
// CONFIGURATION
// ============================================

const CONFIG = {
  provider: process.env.LLM_PROVIDER || "anthropic", // "anthropic" or "openai"
  anthropic: {
    apiKey: process.env.ANTHROPIC_API_KEY,
    model: process.env.ANTHROPIC_MODEL || "claude-sonnet-5",
  },
  openai: {
    apiKey: process.env.OPENAI_API_KEY,
    model: process.env.OPENAI_MODEL || "gpt-5.6-terra",
  },
  maxTokens: 16384,
  // Citation verification burns through searches faster than essay fact-checking did
  maxSearches: Number(process.env.MAX_SEARCHES) || 30,
  // Hard stop on the agentic loop so a misbehaving model can't bill indefinitely
  maxSearchTurns: Number(process.env.MAX_SEARCH_TURNS) || 40,
};

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
const courseBrief = (inputs) => `COURSE BRIEF
Subject: ${inputs.subject}
Level: ${inputs.level}
Length: ${inputs.weeks} weeks
Format: ${inputs.format}`;

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
    description: (inputs) => `${courseBrief(inputs)}

Survey the field so a course can be built on top of your findings.

YOU HAVE ACCESS TO WEB SEARCH. Use it aggressively. Search repeatedly with different queries.
Look specifically for real course syllabi, university catalog listings, review articles,
recent scholarship, and professional or accreditation standards where they apply.

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
    description: (inputs, results) => `${courseBrief(inputs)}

YOU HAVE ACCESS TO WEB SEARCH. Use it heavily. Your primary job is catching citations that
do not exist, and you cannot do that from memory.

Course outline:

${results.outline}

Course materials:

${results.materials}

Underlying research:

${results.research}

Review this course the way a curriculum committee would, but do the citation work a committee
never has time for.

## 1. CITATION VERIFICATION
Go through EVERY cited work in the outline and materials. Search for each one.
Present results as a table with columns: Cited As | Status | Correction / Notes

Use exactly these status values:
- VERIFIED - exists as cited
- CORRECTED - exists, but a detail was wrong (give the corrected citation in full)
- UNVERIFIABLE - could not confirm either way (say what you searched)
- LIKELY FABRICATED - strong evidence this work does not exist

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

function createClient() {
  if (CONFIG.provider === "anthropic") {
    return new Anthropic({ apiKey: CONFIG.anthropic.apiKey });
  } else {
    return new OpenAI({ apiKey: CONFIG.openai.apiKey });
  }
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
async function callLLMSimple(client, agent, prompt) {
  const systemPrompt = buildSystemPrompt(agent);

  if (CONFIG.provider === "anthropic") {
    const response = await client.messages.create({
      model: CONFIG.anthropic.model,
      max_tokens: CONFIG.maxTokens,
      system: systemPrompt,
      messages: [{ role: "user", content: prompt }],
    });
    // Extract text from response, handling different content block types
    const textBlocks = response.content.filter((block) => block.type === "text");
    if (textBlocks.length === 0) {
      throw new Error(`No text in response. Got: ${JSON.stringify(response.content.map(b => b.type))}`);
    }
    return textBlocks.map((b) => b.text).join("\n");
  } else {
    const response = await client.chat.completions.create({
      model: CONFIG.openai.model,
      max_completion_tokens: CONFIG.maxTokens,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: prompt },
      ],
    });
    return response.choices[0].message.content;
  }
}

/**
 * Call Anthropic with web search capability
 * Uses an agentic loop to handle tool calls
 */
async function callAnthropicWithSearch(client, agent, prompt, verbose = true) {
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
      model: CONFIG.anthropic.model,
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
      return textBlocks.map((b) => b.text).join("\n");
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
      return textBlocks.map((b) => b.text).join("\n");
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
async function callOpenAIWithSearch(client, agent, prompt, verbose = true) {
  const systemPrompt = buildSystemPrompt(agent, true);

  const fullPrompt = `${systemPrompt}\n\n${prompt}`;

  if (verbose) {
    console.log(`  [Web Search enabled via OpenAI]`);
  }

  const response = await client.responses.create({
    model: CONFIG.openai.model,
    tools: [{ type: "web_search" }],
    input: fullPrompt,
  });

  return response.output_text;
}

/**
 * Call LLM with web search capability
 * Routes to appropriate provider implementation
 */
async function callLLMWithSearch(client, agent, prompt, verbose = true) {
  if (CONFIG.provider === "anthropic") {
    return callAnthropicWithSearch(client, agent, prompt, verbose);
  } else {
    return callOpenAIWithSearch(client, agent, prompt, verbose);
  }
}

/**
 * Main LLM call function - routes to appropriate handler
 */
async function callLLM(client, agent, prompt, useWebSearch = false, verbose = true) {
  if (useWebSearch) {
    return callLLMWithSearch(client, agent, prompt, verbose);
  }
  return callLLMSimple(client, agent, prompt);
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
  --no-visuals       Skip the Palette agent
  -h, --help         Show this message

Examples:
  node curriculum-squad.js "Media Archaeology"
  node curriculum-squad.js "Mycology for Artists" --level=grad --weeks=10 --format=studio
  node curriculum-squad.js "Data Ethics" --weeks=12 --no-visuals`;

function parseArgs(args) {
  const flags = {
    noVisuals: false, // --no-visuals: skip palette
    help: false,
    subject: null,
    level: DEFAULTS.level,
    weeks: DEFAULTS.weeks,
    format: DEFAULTS.format,
  };

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
    } else if (arg.startsWith("-")) {
      throw new Error(`Unknown option: ${arg}`);
    } else {
      flags.subject = arg;
    }
  }

  return flags;
}

// ============================================
// TEAM RUNNER
// ============================================

async function runTeam(inputs, options = {}) {
  const { verbose = true, onTaskStart, onTaskComplete, skipVisuals = false } = options;
  const client = createClient();
  const results = {};

  // Determine which tasks to run
  const activeTasks = tasks.filter((task) => {
    if (task.optional === "visuals" && skipVisuals) return false;
    return true;
  });

  console.log(`\n${"=".repeat(60)}`);
  console.log(`CURRICULUM DESIGN TEAM`);
  console.log(`Provider: ${CONFIG.provider.toUpperCase()}`);
  console.log(`Subject: ${inputs.subject}`);
  console.log(`Level:   ${inputs.level}`);
  console.log(`Length:  ${inputs.weeks} weeks`);
  console.log(`Format:  ${inputs.format}`);
  if (skipVisuals) console.log(`Mode:    NO VISUALS (skipping palette)`);
  console.log(`${"=".repeat(60)}\n`);

  for (const [index, task] of activeTasks.entries()) {
    const agent = agents[task.agent];
    const taskPrompt = task.description(inputs, results);

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
      console.log(`Web search: ENABLED`);
    }
    const result = await callLLM(client, agent, taskPrompt, useSearch, verbose);
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
function buildProposal(inputs, results) {
  const sections = [
    `# Course Proposal: ${inputs.subject}`,
    `**Level:** ${inputs.level}  \n**Length:** ${inputs.weeks} weeks  \n**Format:** ${inputs.format}  \n**Generated:** ${new Date().toISOString().slice(0, 10)}`,
    `> Assembled by an LLM pipeline. Review the verification report before circulating -\n> citations flagged UNVERIFIABLE or LIKELY FABRICATED still need a human check.`,
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

  const subject = flags.subject || "The cultural history of maps and mapmaking";
  if (!flags.subject) {
    console.log(`No subject given, using default: "${subject}"`);
    console.log(`Run with --help to see options.\n`);
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

    await fs.writeFile(
      `${outputDir}/course-proposal.md`,
      buildProposal(inputs, results)
    );

    console.log(`\nOutputs saved to: ${outputDir}/`);
    console.log(`Start with: ${outputDir}/course-proposal.md`);
    console.log(`Then read:  ${outputDir}/${OUTPUT_FILES.verify} (citation checks)`);
  } catch (error) {
    console.error("\nError running team:", error.message);

    // Don't litter the cwd with an empty directory if we failed before any output
    const written = await fs.readdir(outputDir).catch(() => []);
    if (written.length === 0) {
      await fs.rmdir(outputDir).catch(() => {});
    } else {
      console.error(`Partial results saved in: ${outputDir}/`);
      console.error(`Completed steps: ${written.sort().join(", ")}`);
    }
    process.exit(1);
  }
}

// Run only when executed directly, so importing this file doesn't kick off a run
import { fileURLToPath } from "url";
import { realpathSync } from "fs";

const isDirectRun =
  process.argv[1] &&
  realpathSync(process.argv[1]) === realpathSync(fileURLToPath(import.meta.url));

if (isDirectRun) {
  main();
}

// Export for use as module
export { agents, tasks, runTeam, parseArgs, buildProposal, CONFIG };
