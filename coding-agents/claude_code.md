# Claude Code

## Quick start

- https://code.claude.com/docs/en/quickstart
- Each message, tool use, and result is written to ~/.claude/projects/
  which enables rewinding, resuming, and forking sessions.
- Usage: /usage, /cost, /usage-credits
- Context window: /context, /compact
- Trusted commands: .claude/settings.json
- /init (creates CLAUDE.md), /agents, /doctor

## Extend Claude Code

- CLAUDE.md adds persistent context Claude sees every session
- Skills add reusable knowledge and invocable workflows
- Code intelligence connects Claude to a language server for symbol-level navigation and live type errors
- MCP connects Claude to external services and tools
- Subagents run their own loops in isolated context, returning summaries
- Agent teams coordinate multiple independent sessions with shared tasks and peer-to-peer messaging
- Hooks fire on lifecycle events and can run a script, HTTP request, prompt, or subagent
- Plugins and marketplaces package and distribute these features

