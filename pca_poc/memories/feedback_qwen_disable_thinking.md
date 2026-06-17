---
name: Disable Qwen thinking via --reasoning off
description: Qwen3 thinking mode is disabled with --reasoning off in llama-cli
type: feedback
originSessionId: 0bb1b394-b1bf-4d19-a7a4-c47efd868285
---
To disable thinking mode in Qwen3/Qwen3.5 models with llama-cli, use:

```
--reasoning off
```

**Why:** User first suggested `--chat-template-kwargs "{\"enable_thinking\": false}"` then actually used `--reasoning off` in their test runs. The latter is the simpler, supported flag — `--reasoning` takes a value (`off`/`medium`/`high`) and controls the thinking behavior end-to-end.

**How to apply:** Whenever a Qwen3-family model needs no-thinking mode in llama-cli, append `--reasoning off`. Don't reach for `--chat-template-kwargs` first.
