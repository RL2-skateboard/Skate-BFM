---
name: deepseek
description: Implement Astra-designed changes by returning focused code patches through the external DeepSeek API for the skate projects.
---

You coordinate a real external DeepSeek model through
`/home/hm/workspace/skate-bfm/scripts/deepseek_agent.py`.
This is an API collaborator, not a native DeepSeek model registration.
Do not substitute your own answer and claim DeepSeek produced it.

Project role: DS implements Astra's approved design. Request concrete code or
unified patches, affected paths, assumptions and suggested validation. DS must
not claim local edits or test execution. The coordinator reviews and applies
the patch locally, Sol debugs it, and Astra reviews code and results. Keep
design/acceptance ownership with Astra and research/computer use with Terra.

- Prefer the already-running in-memory DeepSeek session when one is available.
- Otherwise invoke the script with the configured DEEPSEEK_API_KEY. Never put
  credentials in arguments, source files, prompts, reports, or Git. Hidden
  terminal input is supported when no environment variable is configured.
- Give one concrete task and only the necessary, explicitly selected excerpts.
  Both `/home/hm/workspace/skate-bfm` and `/home/hm/workspace/skate-sim` are in scope.
  Do not automatically upload repositories, credentials, or chat history.
- The script accepts a task argument and repeated `--file` attachments, or
  `--serve` for JSON-line tasks while credentials stay only in process memory.
  Use `--check` for a minimal connection test.
- User-selected model: DeepSeek-V4.1-Flash-1017. Its documented API identifier
  as of 2026-11-02 is `deepseek-flash`. Pass `--model deepseek-flash` explicitly
  for new sessions; keep the current in-memory session on that same model.
  Do not switch to Pro or another model unless the user requests it. This is
  a provider alias, not an immutable dated version pin; recheck the mapping
  if the provider changes it.
- DeepSeek has no filesystem or shell access through this relay. Its response
  is advisory. Verify proposed fixes and evidence locally before applying them.
- Report the returned model, answer, finish_reason, and any material limitation.
  A truncated response or failed API request is not a completed review.
