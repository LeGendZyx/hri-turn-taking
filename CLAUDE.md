# CLAUDE.md — Project Context

> The canonical agent guide for this repository is **[AGENTS.md](AGENTS.md)** —
> project overview, research framing, the Key Design Decisions you must not change
> without asking, the repo map, build/test/run commands, conventions, and
> guardrails all live there. It is imported in full below; **edit AGENTS.md (the
> single source of truth), not this file, for shared guidance.**

@AGENTS.md

## Claude Code specifics

- **Language:** respond to the user in **Simplified Chinese (简体中文)** by default;
  keep code, commands, and error messages in English.
- Record durable project context (decisions not derivable from the code or git
  history) in the persistent file-based memory indexed by `MEMORY.md`.
- Background jobs must isolate file edits in a git worktree (`EnterWorktree`)
  before changing repository files; read-only work stays in place.
- Before changing any item in **AGENTS.md §4 (Key Design Decisions)**, ask first.
