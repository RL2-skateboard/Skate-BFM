---
name: deepseek
description: Generate implementation patches for skate-sim from Astra's design through the shared external DeepSeek API relay.
---

Use `/home/hm/workspace/skate-bfm/scripts/deepseek_agent.py` as the external
DeepSeek collaborator for this workspace. Read the shared instructions at
`/home/hm/workspace/skate-bfm/.cursor/agents/deepseek.md` before invoking it.
This is an API relay, not a native DeepSeek model registration. Never claim a
DeepSeek response without actually invoking the API. Keep credentials out of
files, arguments, prompts, output, and Git. Review its proposals locally before
applying changes. It receives only explicitly supplied task text and context.

DS owns implementation proposals; the coordinator applies reviewed patches
locally, Sol handles debugging, and Astra reviews code and actual results.
Request patches with affected paths and suggested validation. DS cannot claim
to have edited files or run checks through this text-only relay.

The user selected DeepSeek-V4.1-Flash-1017. Use its current documented API
identifier `deepseek-flash` explicitly; do not switch models without a user request.
