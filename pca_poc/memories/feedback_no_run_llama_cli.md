---
name: Don't run llama-cli yourself
description: User runs llama-cli commands themselves; only provide the command text for them to paste
type: feedback
originSessionId: 0bb1b394-b1bf-4d19-a7a4-c47efd868285
---
Do not invoke `llama-cli` (or similar end-to-end inference CLIs) via the Bash tool. Only hand the user the command to run.

**Why:** User said "I have told you, not to run anything. Only i run the commands for llama-cli." These runs are long, interactive, and the user wants to watch the output themselves — they will paste results back.

**How to apply:** When a test run is needed, emit the Windows cmd command (with `^` line continuation if multi-line) and stop. Building, python converters, git, grep, etc. are still fine to run via tools. The restriction is specifically about running the inference binary.
