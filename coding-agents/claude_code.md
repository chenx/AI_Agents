# Claude Code

## Quick start

- https://code.claude.com/docs/en/quickstart
- Each message, tool use, and result is written to ~/.claude/projects/
  which enables rewinding, resuming, and forking sessions.
- Usage: /usage, /cost, /usage-credits
- Context window: /context, /compact
- Trusted commands: .claude/settings.json
- /init (creates CLAUDE.md), /agents, /doctor

## Core concepts

### How Claude Code works

- The agentic loop: Models, Tools
- What Claude can access:
  - project, terminal, git state, CLAUDE.md, auto memory
  - extensions:
    - MCP servers for external services,
    - skills for workflows,
    - subagents for delegated work, and
    - Claude in Chrome for browser interaction.

- Environments and interfaces
  - Execution environments: local, cloud, remote control
  - Interfaces: desktop app, IDE extensions, claude.ai/code, Remote Control, Slack, and CI/CD pipelines. 

- Work with sessions
  - Work across branches
  - Resume or fork sessions
  - The context window

- Stay safe with checkpoints and permissions
  - Undo changes with checkpoints
  - Control what Claude can do

- Tips: Work effectively with Claude Code
  - Ask Claude Code for help
  - It’s a conversation
  - Interrupt and steer
  - Be specific upfront
  - Give Claude something to verify against
  - Explore before implementing
  - Delegate, don’t dictate


### Extend Claude Code

- CLAUDE.md adds persistent context Claude sees every session
- Skills add reusable knowledge and invocable workflows
- Code intelligence connects Claude to a language server for symbol-level navigation and live type errors
- MCP connects Claude to external services and tools
- Subagents run their own loops in isolated context, returning summaries
- Agent teams coordinate multiple independent sessions with shared tasks and peer-to-peer messaging
- Hooks fire on lifecycle events and can run a script, HTTP request, prompt, or subagent
- Plugins and marketplaces package and distribute these features

### The .claude directory

- ~/.claude
  - ~/.claude/projects/
  - ~/.claude/settings.json

### Explore the context window

### Prompt caching


## Use Claude Code

### Store instructions and memories

- How Claude remembers your project between sessions:
  - 1) CLAUDE.md, 2) Auto memory
  - loaded at the start of every session
- CLAUDE.md files
  - Where
    - Managed policy:
      • macOS: /Library/Application Support/ClaudeCode/CLAUDE.md
      • Linux and WSL: /etc/claude-code/CLAUDE.md
    - User instructions: ~/.claude/CLAUDE.md
    - Project instructions: ./CLAUDE.md or ./.claude/CLAUDE.md
    - Local instructions: ./CLAUDE.local.md
  - Set up a project CLAUDE.md
  - Write effective instructions:
    - size: target under 200 lines per CLAUDE.md file.
    - structure, specificity, consistency
    - Import additional files: @path/to/import
  - Claude Code reads CLAUDE.md, not AGENTS.md
    - import AGENTS.md from other agents: @AGENTS.md
    - or: ln -s AGENTS.md CLAUDE.md
  - Organize rules with .claude/rules/
  - Manage CLAUDE.md for large teams
  - My CLAUDE.md is too large: /compact
- Auto memory
  - Auto memory lets Claude accumulate knowledge across sessions without you writing anything. 
  - Storage location
    - Each project gets its own memory directory at ~/.claude/projects/&lt;project>/memory/
    - The first 200 lines of MEMORY.md, or the first 25KB, whichever comes first, are loaded at the start of every conversation.
  - /memory


### Permission modes

| Mode                | What runs without asking                                                               | Best for                                |
| ------------------- | -------------------------------------------------------------------------------------- | --------------------------------------- |
| `default`           | Reads only                                                                             | Getting started, sensitive work         |
| `acceptEdits`       | Reads, file edits, and common filesystem commands (`mkdir`, `touch`, `mv`, `cp`, etc.) | Iterating on code you're reviewing      |
| `plan`              | Reads only                                                                             | Exploring a codebase before changing it |
| `auto`              | Everything, with background safety checks                                              | Long tasks, reducing prompt fatigue     |
| `dontAsk`           | Only pre-approved tools                                                                | Locked-down CI and scripts              |
| `bypassPermissions` | Everything                                                                             | Isolated containers and VMs only        |


### Manage sessions

### Common workflows

### Prompt library

### Best practices
