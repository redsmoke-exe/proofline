# ATS CV Agent

A production-oriented Python system that ingests a job description plus multiple candidate
CVs, builds an evidence-backed master profile, aligns truthful content to the role, runs a
strict quality gate, and renders a resume and cover letter through a fixed single-column PDF
template.

The implementation supports OpenAI, Gemini, and OpenRouter behind one typed agent boundary.
OpenAI uses the Responses API with `store=False`; Gemini uses Google's OpenAI-compatible Chat
Completions parser. OpenRouter models use forced schema tool calls followed by strict local
Pydantic validation. When both Gemini and OpenRouter keys are present, independent requests are
hedged and the first schema-valid response wins. Candidate inputs are never written to logs.

## What is enforced

- `.txt`, `.md`, `.pdf`, and `.docx` ingestion; JDs may also be supplied as raw text.
- Multi-CV aggregation with deduplicated roles and exact source-quote provenance.
- Strict Pydantic v2 models with `extra="forbid"` at every pipeline boundary.
- Flexible achievement components (action, object, method, scope, outcome, and verified metrics)
  represented explicitly without forcing every source claim into one formula.
- Candidate identity fields copied exactly from the master profile.
- Candidate skills selected only from source-backed profile skills.
- Numeric claims rejected if the number is absent from cited evidence.
- Exact evidence quotes verified against normalized source documents.
- Supported-requirement coverage, standard headings, parseable links, and single-column template
  checks; metrics are used only when the source verifies a useful number.
- Deterministic career-stage and market strategy for section order, one/two-page budgets, and
  A4/Letter output without turning job requirements into candidate claims.
- Optional summaries and sections, categorized skills, richer projects/certifications/education,
  and evidence-gap questions for information the candidate may want to add later.
- Independent structured groundedness audit plus deterministic local groundedness checks.
- At most two targeted document correction attempts; failure is terminal and no final PDF is
  accepted.
- Stable Jinja templates, fixed CSS, atomic PDF writes, normalized PDF metadata, and SHA-256
  artifact manifests.
- JSON logs for pipeline timing, API latency, token usage, retries, and artifact production.

"Zero hallucination" is implemented as a fail-closed engineering invariant: the system requires
valid provenance IDs, exact source excerpts, unchanged identity fields, source-supported numbers,
lexical support, and an independent semantic audit. It does not claim that an LLM by itself is a
mathematical proof of truth.

## Architecture

```text
CVs/JD -> UniversalDocumentReader -> IngestionBundle
             |
             +----> Profile aggregation ----> exact-evidence gate --+
             |          Gemini | OpenRouter (first valid wins)       |
             |                                                    join
             +----> JD requirement extraction ----------------------+
                        Gemini | OpenRouter (first valid wins)
             |
             v
      deterministic resume strategy (career stage, market, section/page budget)
             |
             v
      Resume + cover letter structured draft
             Gemini | OpenRouter (first valid wins)
             |
             v
  deterministic ATS/provenance gate + semantic groundedness audit
             |
       pass  |  fail (maximum two targeted revisions)
             v
  fixed HTML/CSS -> WeasyPrint -> normalized PDF + audit snapshot
```

Pipeline hooks receive typed state after every stage, so vector retrieval, tracing, a UI, or an
application-submission adapter can be added without coupling those features to ingestion or
rendering.

## Project layout

```text
.
├── main.py                         # source-checkout convenience entrypoint
├── pyproject.toml                  # package metadata and `cv-agent` console script
├── requirements.txt                # fully pinned runtime/test dependencies
├── examples/
│   ├── candidate_history.txt
│   ├── candidate_primary.md
│   ├── job_description.md
│   └── sample_snapshot.json        # validated, offline rendering example
├── src/cv_agent/
│   ├── agent.py                    # prompts, Structured Outputs, API backoff
│   ├── api.py                      # FastAPI contract and unique PDF downloads
│   ├── document_repair.py          # deterministic source-backed document projection
│   ├── ingestion.py                # universal readers and normalization
│   ├── logging_config.py           # JSON observability
│   ├── main.py                     # CLI
│   ├── pipeline.py                 # typed state machine and correction loop
│   ├── renderer.py                 # deterministic HTML/PDF engine
│   ├── schemas.py                  # complete Pydantic v2 contracts
│   ├── strategy.py                 # career-stage, market, section, and page planning
│   ├── validation.py               # ATS and truthfulness gates
│   └── templates/
│       ├── cover_letter.html
│       ├── resume.html
│       └── styles.css
└── tests/                          # ingestion, schema, gate, retry, pipeline, PDF tests
```

## Install

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

WeasyPrint also requires native Pango/Cairo libraries. On macOS:

```bash
brew install pango
```

On Debian/Ubuntu, install the platform packages recommended by WeasyPrint (package names vary by
release), commonly including Pango, Cairo, GDK-PixBuf, and shared MIME data.

Copy `.env.example` values into your shell or secret manager. Do not commit API keys.

```bash
export LLM_PROVIDER="openrouter"
export LLM_PARALLEL_PROVIDERS="true"
export OPENROUTER_API_KEY="..."
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export OPENROUTER_MODEL="nvidia/nemotron-3.5-lightning:free"
export OPENROUTER_FALLBACK_MODELS="openrouter/free"
export GEMINI_API_KEY="..."
export GEMINI_MODEL="gemini-3-flash-preview"
export GEMINI_FALLBACK_MODELS="gemini-3.7-flash,gemini-3.6-flash"
```

The model provider is configurable. `LLM_PROVIDER=openrouter` uses the OpenAI SDK with the
OpenRouter URL above. Fast mode uses `OPENROUTER_REASONING_EFFORT=none`; change it to `minimal`
or higher when reasoning quality matters more than latency. The configured free Nemotron
model supports tool calling but not `response_format`, so the adapter forces a schema tool call
and validates its arguments locally before any pipeline stage accepts them. If the free endpoint
returns schema JSON in message content instead of making the requested tool call, the same strict
local validation handles that response. `OPENROUTER_FALLBACK_MODELS` is a comma-separated ordered
list; `openrouter/free` delegates to OpenRouter's free-model router if the pinned Nemotron endpoint
fails. The retry scheduler tries each configured model once before repeating an unhealthy endpoint.
Safe per-schema output-token caps prevent runaway generation, and `OPENROUTER_MAX_TOKENS` can
override those caps globally. Provider timeouts are surfaced separately from malformed structured
responses.

With `LLM_PARALLEL_PROVIDERS=true`, candidate-profile extraction and JD analysis run concurrently,
and each LLM stage races Gemini against OpenRouter. The losing synchronous request is detached from
the response path and remains bounded by its provider timeout. If one provider fails or emits
invalid schema output, the other can still complete the stage; the API returns a multi-provider
error only when every configured provider fails.

Validated master profiles and JD analyses are cached by a versioned SHA-256 content fingerprint
under `output/cache/`. Cache entries survive backend restarts, are revalidated before use, and are
written atomically with mode `0600` beneath mode-`0700` directories. Set `CV_AGENT_CACHE_DIR` to
move the cache. Editing a CV creates a new fingerprint and therefore a new extraction; switching
back to an earlier unchanged CV version reuses its existing cache entry.

Generated document fields are projected back onto exact, validated master-profile records before
rendering, so a model cannot change names, employers, dates, skills, claims, or metrics. The hard
groundedness audit is deterministic and local. The slower model-based second opinion is advisory
and disabled by default; set `ENABLE_EXTERNAL_GROUNDEDNESS_AUDIT=true` to enable it without making
provider timeouts or auditor disagreements block PDF generation.

`LLM_PROVIDER=gemini` uses Google's OpenAI-compatible Chat Completions parser with Gemini
structured outputs. `LLM_PROVIDER=openai` keeps the original OpenAI Responses API path; set the
corresponding provider key and model variables for either mode. Reasoning-model families such as
GPT-5.6, GPT-6, and Gemini 3 reject or deprecate custom sampling temperatures, so the client omits
`temperature` for those models while retaining it for compatible models. On repeated transient
Gemini `429` or `5xx` failures, the client moves through `GEMINI_FALLBACK_MODELS` in order and logs
each model switch.

## Run

### Start the HTTP API for the React studio

The web application calls the real Python pipeline through a local FastAPI service. The service
loads `LLM_PROVIDER`, provider API keys, and model settings from the repository `.env` file when
present.

```bash
.venv/bin/cv-agent-api
```

The API listens on `http://127.0.0.1:8000`; health status is available at `/api/health` and
interactive API documentation at `/api/docs`. Generated files are isolated by run under
`output/runs/` and are exposed only through allowlisted artifact routes. The UI remains PDF-only;
each download receives a unique `Content-Disposition` filename so repeated downloads do not
silently overwrite one another.

In a second terminal, start the UI:

```bash
cd web
npm run dev
```

Then open `http://localhost:5173`.

### Command-line generation

Run the complete pipeline with any number of CV sources:

```bash
.venv/bin/python main.py generate \
  --jd examples/job_description.md \
  --cv examples/candidate_primary.md \
  --cv examples/candidate_history.txt \
  --output-dir output/pdf \
  --letter-date 2026-09-12
```

Use a raw JD instead of a file:

```bash
.venv/bin/python main.py generate \
  --jd-text "Data Platform Engineer needed ..." \
  --cv candidate.pdf
```

Inspect parsing without making an API call:

```bash
.venv/bin/python main.py inspect \
  --jd examples/job_description.md \
  --cv examples/candidate_primary.md \
  --cv examples/candidate_history.txt
```

Render the provided validated offline example:

```bash
.venv/bin/python main.py render \
  --snapshot examples/sample_snapshot.json \
  --output-dir output/pdf
```

Successful command-line generation writes:

- `resume.pdf` and `cover_letter.pdf`
- matching HTML previews
- `pipeline_snapshot.json`, mode `0600`, containing the full typed audit record

The snapshot and `output/cache/` contain candidate personal data. Retain, encrypt, and delete them
according to your privacy policy.

## Quality-gate behavior

Truth and rendering checks are deliberately strict:

- groundedness: `1.00`
- invented skills, facts, cross-role citations, identity drift, and unsupported metrics: blocked
- ATS-hostile or unsafe templates: blocked

Supported-requirement coverage (`0.65` target), action-verb quality, and bullet clarity are
advisory diagnostics. Verified metric usage is reported as telemetry, not treated as a quota:
truthful qualitative outcomes remain valid. The UI's **Application alignment** score combines
supported-requirement coverage, evidence strength, parseability, and content completeness; it is
an internal document-readiness indicator, not a promise about an employer's ATS or hiring result.
Generated packages are repaired locally in one pass instead of repeatedly sending the full CV
back to the provider.

## Test

```bash
.venv/bin/python -m pytest
```

The suite includes real DOCX/PDF extraction, strict-schema conversion, transient and malformed
API-response recovery, unsupported-metric rejection, deterministic package repair, advisory audit
failure recovery, both API PDF downloads, deterministic template checks, real PDF rendering, and
post-render text extraction.

## Production notes

- Pin and evaluate the chosen provider, primary model, and any ordered fallback models against your
  own CV corpus. Every fallback switch is emitted as an `llm_fallback` event.
- Run the integration suite in the same container image used for production because font and Pango
  versions influence pagination.
- Keep the template package immutable per release. Increment `TEMPLATE_VERSION` for any CSS or HTML
  change and regression-test representative one-, two-, and three-page resumes.
- Put API keys in a managed secret store, add request-level correlation IDs at the service edge, and
  define retention rules for PDFs and snapshots.
- OCR is intentionally not guessed here. Image-only PDFs fail with "no extractable text"; add a
  vetted OCR adapter as a separate ingestion plugin if that input class is required.
