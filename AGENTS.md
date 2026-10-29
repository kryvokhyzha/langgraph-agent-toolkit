# AGENTS.md

Follow [CLAUDE.md](CLAUDE.md) for the repository's project guidance.

## Coding-agent skills

Task-specific guides are in [skills/README.md](skills/README.md). Read only the
skill that applies to the task. The source files are under `skills/`.
`.agents/skills/` and `.claude/skills/` contain discovery links to those files.

## Technical writing

Write comments, docstrings, and documentation in ASD-STE100 Simplified Technical
English.

- Use short, direct sentences in active voice. State one instruction or fact in
  each sentence. Use simple, consistent words. Do not use idioms, ambiguous
  terms, or unnecessary synonyms.
- Preserve each technical identifier exactly. This includes code symbols, API
  names, file paths, commands, options, environment variables, configuration
  keys, URLs, literal values, casing, and punctuation.
- You may use `gpt-5.6-sol`, `gpt-5.6-terra`, or `gpt-5.6-luna` to rewrite
  existing comments and documentation in this style. Change prose only. Do not
  change technical meaning or program behavior.
