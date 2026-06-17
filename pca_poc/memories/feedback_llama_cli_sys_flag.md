---
name: llama-cli system prompt flag is -sys
description: Use -sys (not --sys-prompt) for system prompt in llama-cli
type: feedback
originSessionId: 0bb1b394-b1bf-4d19-a7a4-c47efd868285
---
For llama-cli, the system prompt flag is `-sys "<text>"`, not `--sys-prompt`.

**Why:** User corrected me when I used `--sys-prompt`. llama.cpp's arg parser accepts the short form `-sys` (and likely the long `--sys-prompt` form too, but the user prefers the short form).

**How to apply:** When constructing llama-cli commands that need a system prompt, use `-sys "<text>"`.
