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

Name, resume, branch, and switch between Claude Code conversations. Covers --continue, --resume, --from-pr, the /resume picker, session naming, and where transcripts are stored.

- Resume a previous conversation by flag, name, or PR
- Name sessions so you can find them later
- Browse sessions with the /resume picker
- Branch a conversation to try a different approach
- Export transcripts and find them on disk


### Common workflows

- Prompt recipes for exploring code, fixing bugs, refactoring, testing, PRs, and documentation
- Resume previous conversations so a task can span multiple sittings
- Run parallel sessions with worktrees so concurrent edits don’t collide
- Plan before editing to review changes before they touch disk
- Delegate research to subagents to keep your main context clean
- Pipe Claude into scripts for CI and batch processing


### Prompt library

### Best practices

- Give Claude a way to verify its work
- Explore first, then plan, then code
- Provide specific context in your prompts
  - Provide rich content
- Configure your environment
- Communicate effectively
- Manage your session
- Automate and scale
- Avoid common failure patterns
- Develop your intuition


## Platforms and integrations

### Overview

#### Where to run Claude Code

| Platform      | Best for                                                                                           | What you get                                                                                                                  |
| ------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **CLI**       | Terminal workflows, scripting, remote servers                                                      | Full feature set, Agent SDK, computer use on macOS (Pro and Max), third-party providers                                       |
| **Desktop**   | Visual review, parallel sessions, managed setup                                                    | Diff viewer, app preview, computer use and Dispatch on Pro and Max                                                            |
| **VS Code**   | Working inside VS Code without switching to a terminal                                             | Inline diffs, integrated terminal, file context                                                                               |
| **JetBrains** | Working inside IntelliJ, PyCharm, WebStorm, or other JetBrains IDEs                                | Diff viewer, selection sharing, terminal session                                                                              |
| **Web**       | Long-running tasks that don’t need much steering, or work that should continue when you’re offline | Anthropic-managed cloud, continues after you disconnect                                                                       |
| **Mobile**    | Starting and monitoring tasks while away from your computer                                        | Cloud sessions from the Claude app for iOS and Android, Remote Control for local sessions, Dispatch to Desktop on Pro and Max |


#### Connect your tools

Integrations let Claude work with services outside your codebase.

| Integration        | What it does                                       | Use it for                                                       |
| ------------------ | -------------------------------------------------- | ---------------------------------------------------------------- |
| **Chrome**         | Controls your browser with your logged-in sessions | Testing web apps, filling forms, automating sites without an API |
| **GitHub Actions** | Runs Claude in your CI pipeline                    | Automated PR reviews, issue triage, scheduled maintenance        |
| **GitLab CI/CD**   | Same as GitHub Actions for GitLab                  | CI-driven automation on GitLab                                   |
| **Code Review**    | Reviews every PR automatically                     | Catching bugs before human review                                |
| **Slack**          | Responds to @Claude mentions in your channels      | Turning bug reports into pull requests from team chat            |


#### Work when you are away from your terminal

Claude Code offers several ways to work when you’re not at your terminal. They differ in what triggers the work, where Claude runs, and how much you need to set up.

| Feature             | Trigger                                                                  | Claude runs on                | Setup                                                     | Best for                                                      |
| ------------------- | ------------------------------------------------------------------------ | ----------------------------- | --------------------------------------------------------- | ------------------------------------------------------------- |
| **Dispatch**        | Message a task from the Claude mobile app                                | Your machine (Desktop)        | Pair the mobile app with Desktop                          | Delegating work while you're away, minimal setup              |
| **Remote Control**  | Drive a running session from `claude.ai/code` or the Claude mobile app   | Your machine (CLI or VS Code) | Run `claude remote-control`                               | Steering in-progress work from another device                 |
| **Channels**        | Push events from a chat app like Telegram or Discord, or your own server | Your machine (CLI)            | Install a channel plugin or build your own                | Reacting to external events like CI failures or chat messages |
| **Slack**           | Mention `@Claude` in a team channel                                      | Anthropic cloud               | Install the Slack app with Claude Code on the web enabled | PRs and reviews from team chat                                |
| **Scheduled tasks** | Set a schedule                                                           | CLI, Desktop, or cloud        | Pick a frequency                                          | Recurring automation like daily reviews                       |


### Remote Control

#### Start a Remote Control session
```
claude remote-control --name [name]
```

- Connect from another device
- Enable Remote Control for all sessions
- Remote Control vs Claude Code on the web
- Mobile push notifications

#### Connection and security


### Claude Code on the web

### Claude Code on desktop

### Chrome extesion (beta)

### Computer use (preview)

### Visual Studio Code

### JetBrains IDEs

### Code Review & CI/CD

### Claude Code in Slack
