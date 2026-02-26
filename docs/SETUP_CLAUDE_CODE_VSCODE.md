# Claude Code + VS Code Setup (macOS)

## Current machine status
- `node`: not installed
- `npm`: not installed
- `claude`: not installed
- `code` (VS Code CLI): not installed

## 1) Install Node.js (required for Claude Code)
Using Homebrew:
```bash
brew install node
```

## 2) Install Claude Code CLI
```bash
npm install -g @anthropic-ai/claude-code
```

## 3) Start Claude Code and authenticate
```bash
claude
```
Follow the login flow in terminal.

## 4) Install VS Code integration from Claude Code
Inside Claude Code, run:
```text
/ide
```
Then choose VS Code from the prompt.

## 5) Enable VS Code command-line launcher (`code`)
In VS Code:
1. Open Command Palette.
2. Run `Shell Command: Install 'code' command in PATH`.

## 6) Open this project in VS Code
```bash
code "/Users/maxwilliams/Library/CloudStorage/Dropbox/MW_GR_data/Data"
```

## Official docs
- https://docs.anthropic.com/en/docs/claude-code/setup
- https://docs.anthropic.com/en/docs/claude-code/ide-integrations
