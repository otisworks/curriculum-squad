/**
 * Test suite for curriculum-squad.
 *
 * Makes NO API calls and costs nothing to run:
 *
 *   npm test
 *
 * It exercises everything that can be checked without a provider - prompt
 * assembly, provider resolution, argument parsing, error classification, retry
 * policy, and the truncation guard. The prompts themselves are the product, so
 * the checks that matter most are the ones asserting that Critic cannot claim a
 * citation is VERIFIED on a provider that has no web search.
 */

import assert from "node:assert";
import {
  agents,
  tasks,
  parseArgs,
  buildProposal,
  resolveProvider,
  assertComplete,
  explainError,
  withRetry,
  parseEnvFile,
  PROVIDERS,
} from "./curriculum-squad.js";

// ---------------------------------------------------------------
// Minimal harness
// ---------------------------------------------------------------

let passed = 0;
let failed = 0;
const failures = [];

function group(name) {
  console.log(`\n${name}`);
}

async function check(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`  ok    ${name}`);
  } catch (error) {
    failed++;
    failures.push({ name, error });
    console.log(`  FAIL  ${name}`);
    console.log(`        ${error.message.split("\n")[0]}`);
  }
}

/** Run fn with specific env vars set, restoring whatever was there before. */
function withEnv(vars, fn) {
  const saved = {};
  for (const [key, value] of Object.entries(vars)) {
    saved[key] = process.env[key];
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  try {
    return fn();
  } finally {
    for (const [key, value] of Object.entries(saved)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
}

// ---------------------------------------------------------------

group("Agents and tasks");

await check("five agents, correctly keyed", () => {
  assert.deepStrictEqual(Object.keys(agents), [
    "archive",
    "dean",
    "praxis",
    "critic",
    "palette",
  ]);
});

await check("every agent has a complete persona", () => {
  for (const [key, agent] of Object.entries(agents)) {
    for (const field of ["name", "role", "goal", "background"]) {
      assert.ok(agent[field]?.trim(), `${key}.${field} is missing or empty`);
    }
  }
});

await check("pipeline order is research -> outline -> materials -> verify -> visuals", () => {
  assert.deepStrictEqual(
    tasks.map((t) => t.id),
    ["research", "outline", "materials", "verify", "visuals"]
  );
});

await check("every task points at a real agent", () => {
  for (const task of tasks) {
    assert.ok(agents[task.agent], `task "${task.id}" references unknown agent "${task.agent}"`);
  }
});

await check("web search is requested by research and verify only", () => {
  assert.deepStrictEqual(
    tasks.filter((t) => t.useWebSearch).map((t) => t.id),
    ["research", "verify"]
  );
});

await check("visuals is the only skippable task", () => {
  assert.deepStrictEqual(
    tasks.filter((t) => t.optional).map((t) => t.optional),
    ["visuals"]
  );
});

// ---------------------------------------------------------------

group("Provider registry");

await check("all four providers are registered", () => {
  assert.deepStrictEqual(Object.keys(PROVIDERS), [
    "anthropic",
    "openai",
    "openrouter",
    "deepseek",
  ]);
});

await check("every provider declares the fields the client layer needs", () => {
  for (const [name, provider] of Object.entries(PROVIDERS)) {
    for (const field of ["label", "sdk", "apiKeyEnv", "modelEnv", "defaultModel", "search"]) {
      assert.ok(provider[field], `${name}.${field} is missing`);
    }
    assert.ok(["anthropic", "openai"].includes(provider.sdk), `${name}: unknown sdk`);
    assert.ok(["native", "plugin", "none"].includes(provider.search), `${name}: unknown search mode`);
  }
});

await check("OpenAI-compatible providers carry their own baseURL", () => {
  assert.strictEqual(PROVIDERS.openrouter.baseURL, "https://openrouter.ai/api/v1");
  assert.strictEqual(PROVIDERS.deepseek.baseURL, "https://api.deepseek.com");
  assert.ok(!PROVIDERS.openai.baseURL, "openai should use the SDK default endpoint");
});

await check("deepseek is the only provider without search", () => {
  const withoutSearch = Object.entries(PROVIDERS)
    .filter(([, p]) => p.search === "none")
    .map(([name]) => name);
  assert.deepStrictEqual(withoutSearch, ["deepseek"]);
});

// ---------------------------------------------------------------

group("Provider resolution");

await check("unknown provider names are rejected", () => {
  assert.throws(() => resolveProvider("groq"), /Unknown provider/);
});

await check("a missing key names the exact variable to set", () => {
  withEnv({ DEEPSEEK_API_KEY: undefined }, () => {
    assert.throws(() => resolveProvider("deepseek"), /DEEPSEEK_API_KEY/);
  });
});

await check("resolution returns key, model, and search mode", () => {
  withEnv({ DEEPSEEK_API_KEY: "test-key", DEEPSEEK_MODEL: undefined }, () => {
    const provider = resolveProvider("deepseek");
    assert.strictEqual(provider.apiKey, "test-key");
    assert.strictEqual(provider.model, "deepseek-v4-pro");
    assert.strictEqual(provider.search, "none");
  });
});

await check("the model env var overrides the default", () => {
  withEnv({ DEEPSEEK_API_KEY: "test-key", DEEPSEEK_MODEL: "deepseek-v4-flash" }, () => {
    assert.strictEqual(resolveProvider("deepseek").model, "deepseek-v4-flash");
  });
});

// ---------------------------------------------------------------

group("Truncation guard");

// Regression: a real run once returned a 1312-character materials step, cut off
// mid-table. It was written to disk, stitched into the proposal, and "verified"
// by Critic. Nothing detected it.
const praxis = { name: "Praxis" };

await check("a response stopped for length is rejected", () => {
  assert.throws(
    () => assertComplete(praxis, "x".repeat(50000), "max_tokens"),
    /cut off mid-sentence/
  );
});

await check("the OpenAI-style 'length' finish reason is also caught", () => {
  assert.throws(
    () => assertComplete(praxis, "x".repeat(50000), "length"),
    /cut off mid-sentence/
  );
});

await check("the original 1312-character truncation would now be caught", () => {
  assert.throws(() => assertComplete(praxis, "x".repeat(1312), "end_turn"), /only 1312 characters/);
});

await check("empty and null responses are rejected", () => {
  assert.throws(() => assertComplete(praxis, "", "end_turn"), /only 0 characters/);
  assert.throws(() => assertComplete(praxis, null, "end_turn"), /only 0 characters/);
});

await check("truncation errors are retryable and free of internals", () => {
  try {
    assertComplete(praxis, "x".repeat(50000), "max_tokens");
    assert.fail("should have thrown");
  } catch (error) {
    assert.strictEqual(error.retryable, true);
    assert.ok(
      !/CONFIG\.|maxTokens|MIN_OUTPUT_CHARS/.test(`${error.message}${error.guidance}`),
      "error text exposes source-code internals to the user"
    );
  }
});

await check("a healthy response passes through untouched", () => {
  const output = "y".repeat(30000);
  assert.strictEqual(assertComplete(praxis, output, "end_turn"), output);
});

// ---------------------------------------------------------------

group("Error translation");

const anthropic = withEnv({ ANTHROPIC_API_KEY: "test-key" }, () => resolveProvider("anthropic"));

const errorCases = [
  [{ status: 401, message: "invalid x-api-key" }, "rejected your API key", false],
  [{ status: 403, message: "forbidden" }, "rejected your API key", false],
  [{ status: 404, message: "model not found" }, "doesn't recognise the model", false],
  [{ status: 429, message: "rate limit exceeded" }, "rate limiting", true],
  [{ status: 500, message: "internal server error" }, "trouble on their end", true],
  [{ status: 529, message: "overloaded" }, "trouble on their end", true],
  [{ message: "insufficient credit balance" }, "billing or credit", false],
  [{ message: "fetch failed ENOTFOUND" }, "Couldn't reach", true],
  [{ message: "prompt is too long" }, "too large", false],
];

for (const [error, expectedText, expectedRetryable] of errorCases) {
  const label = `${error.status ?? "no status"}: ${error.message.slice(0, 30)}`;
  await check(label, () => {
    const result = explainError(error, anthropic);
    assert.ok(
      result.headline.includes(expectedText),
      `headline was "${result.headline}"`
    );
    assert.strictEqual(result.retryable, expectedRetryable, "retryable flag is wrong");
    assert.ok(result.guidance.length > 20, "guidance is too thin to act on");
  });
}

await check("unrecognised errors still produce usable guidance", () => {
  const result = explainError(new Error("something bizarre"), anthropic);
  assert.ok(result.headline.length > 0);
  assert.ok(/--debug/.test(result.guidance), "should point at --debug");
});

// ---------------------------------------------------------------

group("Retry policy");

await check("a transient failure is retried and can succeed", async () => {
  let attempts = 0;
  const result = await withRetry(
    async () => {
      attempts++;
      if (attempts === 1) {
        throw Object.assign(new Error("stopped early"), {
          retryable: true,
          guidance: "temporary",
        });
      }
      return "recovered";
    },
    { provider: anthropic, verbose: false, delayMs: 1 }
  );
  assert.strictEqual(attempts, 2);
  assert.strictEqual(result, "recovered");
});

await check("an auth failure is not retried", async () => {
  let attempts = 0;
  await assert.rejects(
    withRetry(
      async () => {
        attempts++;
        throw Object.assign(new Error("invalid x-api-key"), { status: 401 });
      },
      { provider: anthropic, verbose: false, delayMs: 1 }
    )
  );
  assert.strictEqual(attempts, 1, "retrying a bad key just wastes the user's time");
});

await check("a persistent transient failure gives up after one retry", async () => {
  let attempts = 0;
  await assert.rejects(
    withRetry(
      async () => {
        attempts++;
        throw Object.assign(new Error("overloaded"), { status: 529 });
      },
      { provider: anthropic, verbose: false, delayMs: 1 }
    ),
    /overloaded/
  );
  assert.strictEqual(attempts, 2);
});

// ---------------------------------------------------------------

group("Env file parsing");

await check("handles export prefixes, quotes, comments, and blank lines", () => {
  const vars = parseEnvFile(
    [
      "# a comment",
      "",
      'export ANTHROPIC_API_KEY="sk-ant-123"',
      "OPENAI_API_KEY='sk-oai'",
      "LLM_PROVIDER=deepseek  # trailing comment",
      "WEIRD=a=b=c",
      "  SPACED  =  value  ",
      "NOEQUALS",
    ].join("\n")
  );
  assert.strictEqual(vars.ANTHROPIC_API_KEY, "sk-ant-123");
  assert.strictEqual(vars.OPENAI_API_KEY, "sk-oai");
  assert.strictEqual(vars.LLM_PROVIDER, "deepseek");
  assert.strictEqual(vars.WEIRD, "a=b=c");
  assert.strictEqual(vars.SPACED, "value");
  assert.ok(!("NOEQUALS" in vars), "a line without '=' should be skipped");
});

await check("does not mangle a quoted value containing '#'", () => {
  assert.strictEqual(parseEnvFile('KEY="ab#cd"').KEY, "ab#cd");
});

// ---------------------------------------------------------------

group("Argument parsing");

await check("defaults are applied to a bare subject", () => {
  const flags = parseArgs(["Media Archaeology"]);
  assert.strictEqual(flags.subject, "Media Archaeology");
  assert.strictEqual(flags.level, "upper-division undergraduate");
  assert.strictEqual(flags.weeks, 15);
  assert.strictEqual(flags.format, "seminar");
  assert.strictEqual(flags.provider, null);
});

// Regression: this silently produced a course on "Cartography".
await check("an unquoted subject is joined rather than truncated", () => {
  const flags = parseArgs(["History", "of", "Cartography"]);
  assert.strictEqual(flags.subject, "History of Cartography");
  assert.strictEqual(flags.subjectWasUnquoted, true);
});

await check("a quoted subject is not flagged as unquoted", () => {
  assert.ok(!parseArgs(["History of Cartography"]).subjectWasUnquoted);
});

await check("flag values are never absorbed into the subject", () => {
  const flags = parseArgs(["--weeks", "12", "Real Subject"]);
  assert.strictEqual(flags.subject, "Real Subject");
  assert.strictEqual(flags.weeks, 12);
});

await check("both --key=value and --key value forms work", () => {
  assert.strictEqual(parseArgs(["X", "--weeks=10"]).weeks, 10);
  assert.strictEqual(parseArgs(["X", "--weeks", "10"]).weeks, 10);
});

await check("level aliases expand, free text passes through", () => {
  assert.strictEqual(parseArgs(["X", "--level=grad"]).level, "graduate");
  assert.strictEqual(parseArgs(["X", "--level=phd"]).level, "doctoral");
  assert.strictEqual(parseArgs(["X", "--level=MFA candidates"]).level, "MFA candidates");
});

await check("every registered provider is accepted by --provider", () => {
  for (const name of Object.keys(PROVIDERS)) {
    assert.strictEqual(parseArgs(["X", `--provider=${name}`]).provider, name);
  }
});

await check("an unknown provider is rejected", () => {
  assert.throws(() => parseArgs(["X", "--provider=groq"]), /Unknown provider/);
});

await check("invalid week counts are rejected", () => {
  for (const bad of ["0", "-5", "99", "10.5", "abc"]) {
    assert.throws(() => parseArgs(["X", `--weeks=${bad}`]), /whole number/, `accepted "${bad}"`);
  }
});

await check("unknown flags are rejected", () => {
  assert.throws(() => parseArgs(["X", "--raw"]), /Unknown option/);
});

await check("--no-visuals and --debug are recognised", () => {
  assert.strictEqual(parseArgs(["X", "--no-visuals"]).noVisuals, true);
  assert.strictEqual(parseArgs(["X", "--debug"]).debug, true);
});

// ---------------------------------------------------------------

group("Prompt assembly with search available");

const inputs = { subject: "Media Archaeology", level: "graduate", weeks: 10, format: "seminar" };
const searchCtx = { searchAvailable: true };
const searchResults = {};
const searchPrompts = {};

await check("every prompt builds, with all upstream context interpolated", () => {
  for (const task of tasks) {
    const prompt = task.description(inputs, searchResults, searchCtx);
    assert.ok(prompt.length > 200, `${task.id}: prompt is suspiciously short`);
    assert.ok(!prompt.includes("undefined"), `${task.id}: "undefined" leaked into the prompt`);
    assert.ok(!prompt.includes("[object Object]"), `${task.id}: bad interpolation`);
    searchPrompts[task.id] = prompt;
    searchResults[task.id] = `## SIMULATED ${task.id.toUpperCase()} OUTPUT`;
  }
});

await check("course parameters reach every agent", () => {
  for (const [id, prompt] of Object.entries(searchPrompts)) {
    assert.ok(prompt.includes("Media Archaeology"), `${id}: missing subject`);
    assert.ok(prompt.includes("graduate"), `${id}: missing level`);
    assert.ok(prompt.includes("10 weeks"), `${id}: missing length`);
    assert.ok(prompt.includes("seminar"), `${id}: missing format`);
  }
});

await check("each task receives the outputs it depends on", () => {
  assert.ok(searchPrompts.outline.includes("SIMULATED RESEARCH OUTPUT"));
  assert.ok(searchPrompts.materials.includes("SIMULATED RESEARCH OUTPUT"));
  assert.ok(searchPrompts.materials.includes("SIMULATED OUTLINE OUTPUT"));
  assert.ok(searchPrompts.verify.includes("SIMULATED OUTLINE OUTPUT"));
  assert.ok(searchPrompts.verify.includes("SIMULATED MATERIALS OUTPUT"));
  assert.ok(searchPrompts.visuals.includes("SIMULATED OUTLINE OUTPUT"));
  assert.ok(searchPrompts.visuals.includes("SIMULATED MATERIALS OUTPUT"));
});

await check("citation discipline reaches the source-producing agents", () => {
  for (const id of ["research", "materials"]) {
    assert.ok(searchPrompts[id].includes("Never invent a source"), `${id}: missing citation rule`);
  }
});

await check("the full verification vocabulary is offered", () => {
  assert.ok(searchPrompts.verify.includes("VERIFIED - exists as cited"));
  assert.ok(searchPrompts.verify.includes("LIKELY FABRICATED"));
});

// ---------------------------------------------------------------

group("Prompt assembly without search (the DeepSeek path)");

const noSearchCtx = { searchAvailable: false };
const noSearchResults = {};
const noSearchPrompts = {};

await check("every prompt still builds cleanly", () => {
  for (const task of tasks) {
    const prompt = task.description(inputs, noSearchResults, noSearchCtx);
    assert.ok(!prompt.includes("undefined"), `${task.id}: "undefined" leaked into the prompt`);
    noSearchPrompts[task.id] = prompt;
    noSearchResults[task.id] = `## SIMULATED ${task.id.toUpperCase()} OUTPUT`;
  }
});

await check("the search-dependent agents are told they have none", () => {
  assert.ok(noSearchPrompts.research.includes("NO WEB ACCESS"));
  assert.ok(noSearchPrompts.verify.includes("NO WEB ACCESS"));
});

// The single most important assertion in this file. Without search, Critic would
// be checking citations against the same memory that produced them, so agreement
// proves nothing. A confident VERIFIED would launder an unchecked reading list
// into one that looks audited.
await check("Critic cannot claim anything is VERIFIED", () => {
  const prompt = noSearchPrompts.verify;
  assert.ok(
    prompt.includes("The label VERIFIED is unavailable to you"),
    "missing the explicit prohibition"
  );
  assert.ok(prompt.includes("Do not use VERIFIED"), "missing the prohibition in the table spec");
  assert.ok(
    !prompt.includes("VERIFIED - exists as cited"),
    "search-mode vocabulary leaked into the no-search prompt"
  );
  assert.ok(
    !prompt.includes("LIKELY FABRICATED"),
    "search-mode vocabulary leaked into the no-search prompt"
  );
});

await check("a degraded status vocabulary is supplied instead", () => {
  for (const status of ["PLAUSIBLE", "SUSPECT", "UNCHECKABLE"]) {
    assert.ok(noSearchPrompts.verify.includes(status), `missing ${status}`);
  }
});

await check("Critic still performs every check that needs no web access", () => {
  for (const section of [
    "OUTCOME ALIGNMENT",
    "WORKLOAD REALISM",
    "SEQUENCING",
    "ASSESSMENT VALIDITY",
  ]) {
    assert.ok(noSearchPrompts.verify.includes(section), `dropped ${section}`);
  }
});

await check("Archive marks its citations unverified", () => {
  assert.ok(noSearchPrompts.research.includes("[UNVERIFIED]"));
});

await check("agents that never search get identical prompts either way", () => {
  for (const id of ["outline", "materials", "visuals"]) {
    assert.strictEqual(
      searchPrompts[id],
      noSearchPrompts[id],
      `${id} should not vary with search availability`
    );
  }
});

// ---------------------------------------------------------------

group("Proposal assembly");

await check("all sections are present and nothing is undefined", () => {
  const doc = buildProposal(inputs, searchResults, searchCtx);
  for (const heading of [
    "# Course Proposal: Media Archaeology",
    "# Course Outline",
    "# Materials & Assessment",
    "# Visual & Media Plan",
    "# Verification Report",
  ]) {
    assert.ok(doc.includes(heading), `missing "${heading}"`);
  }
  assert.ok(!doc.includes("undefined"));
});

await check("the visuals section is omitted when skipped", () => {
  const { visuals, ...withoutVisuals } = searchResults;
  const doc = buildProposal(inputs, withoutVisuals, searchCtx);
  assert.ok(!doc.includes("# Visual & Media Plan"));
  assert.ok(doc.includes("# Verification Report"));
  assert.ok(!doc.includes("undefined"));
});

await check("a verified run carries the standard provenance note", () => {
  const doc = buildProposal(inputs, searchResults, searchCtx);
  assert.ok(doc.includes("Review the verification report"));
  assert.ok(!doc.includes("NO SOURCES IN THIS DOCUMENT HAVE BEEN VERIFIED"));
});

// The warning has to travel with the file: a console message scrolls away, but
// this document is the thing someone forwards to a colleague.
await check("an unsearched run is stamped as unverified", () => {
  const doc = buildProposal(inputs, noSearchResults, noSearchCtx);
  assert.ok(doc.includes("**NO SOURCES IN THIS DOCUMENT HAVE BEEN VERIFIED.**"));
  assert.ok(doc.includes("Do not circulate as-is"));
});

// ---------------------------------------------------------------

console.log(`\n${"-".repeat(52)}`);
console.log(`${passed} passed, ${failed} failed`);

if (failed > 0) {
  console.log(`${"-".repeat(52)}\n`);
  for (const { name, error } of failures) {
    console.log(`${name}\n${error.stack}\n`);
  }
  process.exit(1);
}

console.log(`${"-".repeat(52)}\n`);
