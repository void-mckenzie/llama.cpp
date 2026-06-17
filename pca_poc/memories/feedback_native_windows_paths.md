---
name: Use Windows paths for native Windows Python/tools
description: kvpress_env Python is native Windows — use C:\ paths, never /c/ or /tmp/
type: feedback
originSessionId: 0bb1b394-b1bf-4d19-a7a4-c47efd868285
---
When invoking the kvpress_env Python (`C:/Users/MukkeshGanesh/miniconda3/envs/kvpress_env/python.exe`) or other native Windows binaries via Bash, use Windows-style paths (`C:/temp/foo.txt` or `C:\\temp\\foo.txt`), not Git-Bash mounts (`/c/temp/foo.txt`, `/tmp/`).

**Why:** Native Windows Python can't resolve Git-Bash's `/c/` mount or `/tmp/` (which is a Git-Bash invention). It will throw `FileNotFoundError`. The user corrected me twice on this — "dudem i told you, use (kvpress_env) for pythonb" and "Dude, ffs, use windows paths".

**How to apply:** Any path passed to a Windows-native binary (Python from conda, dumpbin, etc.) must be `C:\...` or `C:/...`. If you need a temp file, use `C:/Users/MukkeshGanesh/AppData/Local/Temp/` or just `C:/temp/`. Don't redirect to `/tmp` or `/c/temp` when the consumer is native Windows.
