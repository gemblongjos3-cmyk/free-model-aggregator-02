# Free Model Aggregator — free-model-aggregator-02

A single-file, standard-library-only Python server (`main.py`) that acts
as a live aggregator over OpenRouter's free models.

## What it does

- Serves both client wires: OpenAI Chat Completions
  (`POST /v1/chat/completions`) and Anthropic Messages
  (`POST /v1/messages`), in JSON and streaming form.
- Serves `GET /v1/models` in the shape the caller's auth header asks
  for: `Authorization: Bearer` gets the OpenAI list, `x-api-key` gets
  the Anthropic list. Both list exactly the brief's fourteen free
  models.
- Dials every completion through the gateway's Chat Completions route
  only. Anthropic requests are translated into that shape on the way
  in and the answer is translated back on the way out; the `model`
  field of every response and every streamed chunk names the model
  that was actually dialed.
- Prints its provider catalog with `python3 main.py --list` — the same
  document, byte for byte, that it registers with the environment at
  the start of a trial.

## Routing policy

For each prompt, candidates are the free models that meet the prompt's
declared requirements (tools / vision / JSON), ordered by the house
rating, best first. A refusal is retried once on the same model; a
model that refuses twice in one trial is skipped for the rest of that
trial; otherwise the router fails over to the next candidate. Refusals
that a served prompt recovered from are reported on the response in
the `x-makefun-upstream-failures` header. If every candidate refuses,
the client gets a 503 in its own wire's error shape, with the failure
trail attached.

Streaming answers are buffered, parsed, and re-emitted synthesized in
the client's wire (OpenAI SSE ending in a usage chunk and `[DONE]`;
Anthropic SSE with `message_start` / content blocks / `message_delta` /
`message_stop` and mapped stop reasons), with usage carried whole.

Everything emitted is a deterministic function of the brief, the
request, and the gateway's answers: fixed timestamps, counter-based
ids, no wall clock, no randomness — so a replay of a trial reproduces
the same state export byte for byte.

## Running

The trial harness sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`,
`OPENAI_BASE_URL`, and `OPENAI_API_KEY`. The server binds a free port
on `0.0.0.0`, registers its port and catalog with the environment,
polls `/work`, and exits 0 when the environment reports the trial
done. Outside a trial, `--list` prints the catalog (with an empty model
list when no brief is present) and exits 0.
