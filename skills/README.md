# Coding-agent skills

These skills help coding agents work with `langgraph-agent-toolkit`. They use
the [Agent Skills format](https://agentskills.io/specification). Each folder
contains a `SKILL.md` with a name, a task description, and instructions.
Conditional details live in that skill's `references/` directory.

These files guide development. They do not configure the agents served by the
API or install skills into a Deep Agents runtime backend.

## Choose a skill

| Skill                                                                   | Use it for                                                                                               |
| ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| [langgraph-toolkit-agent](langgraph-toolkit-agent/SKILL.md)             | Build and register application agents. Add MCP, Deep Agents, or tool selection when needed.              |
| [langgraph-toolkit-client](langgraph-toolkit-client/SKILL.md)           | Integrate the HTTP API or Python client. Handle identity, streams, pagination, feedback, and interrupts. |
| [langgraph-toolkit-reliability](langgraph-toolkit-reliability/SKILL.md) | Diagnose persistence, concurrency, connection failures, and worker recovery.                             |
| [langgraph-toolkit-maintainer](langgraph-toolkit-maintainer/SKILL.md)   | Change the toolkit package. Preserve public contracts and choose regression tests.                       |

Select the skill that matches the task. A task can use more than one skill. The
skills do not require loading the full collection.

## Use this checkout

`skills/` is the source directory. This repository includes relative links from
`.agents/skills/<name>` and `.claude/skills/<name>` to each source folder. Edit
the source files under `skills/`. There are no duplicate instruction files.

Codex discovers repository skills through `.agents/skills`. Claude Code uses
`.claude/skills` for local project skills. Start the coding agent in this
checkout. If the new skills do not appear, refresh its skill list or start a new
session. Other clients can use their supported installation path.

Example Codex prompts:

```text
Use $langgraph-toolkit-agent to add a native agent with an async search tool.
Use $langgraph-toolkit-client to add streaming and paginated history to my app.
Use $langgraph-toolkit-reliability to investigate history loss after worker restart.
Use $langgraph-toolkit-maintainer to fix this API bug and test its public contract.
```

In Claude Code, invoke the same names with `/`, such as
`/langgraph-toolkit-client`. Skill selection can also use the task description.

Git must preserve symbolic links for this checkout layout. On systems that check
them out as text files, install copies in the destination project with the
command below. Do not maintain a second edited copy in this repository.

## Install in another project

The [skills CLI](https://github.com/vercel-labs/skills) supports local folders
and Git repositories. CLI version 1.7.4 requires Node.js 22.20 or later. The
manual copy option below does not need Node.js. Run the CLI from the destination
project. Preview the local collection first:

```sh
npx skills add /path/to/langgraph-agent-toolkit/skills --list
```

Install the selected skills for Codex:

```sh
npx skills add /path/to/langgraph-agent-toolkit/skills \
  --skill langgraph-toolkit-agent langgraph-toolkit-client --agent codex
```

Use `--agent claude-code` for Claude Code. Use `--copy` when symbolic links are
not available. For example, from another project's root:

```sh
npx skills add /path/to/langgraph-agent-toolkit/skills \
  --skill langgraph-toolkit-client --agent claude-code --copy
```

After these files are published to GitHub, the repository form is:

```sh
npx skills add kryvokhyzha/langgraph-agent-toolkit \
  --skill langgraph-toolkit-agent langgraph-toolkit-client --agent codex
```

Without this CLI, copy each selected skill folder into the coding agent's
supported skills directory. Include its `references/` directory. The skills have
no dependency on neighboring skill folders or this repository's absolute path.
They refer to the installed toolkit version and link to public sources.

Skills are distributed from the repository. Installing the Python package does
not register these skills in a coding agent.

## Maintain the collection

Keep the YAML `name` equal to the folder name. Make `description` identify the
task that should select the skill. Put the main workflow in `SKILL.md`. Link
conditional detail from it. Add scripts or templates only when they remove
repeated work that instructions cannot handle well.

Keep project-wide rules in `CLAUDE.md`. `AGENTS.md` points to that file. Keep
full product documentation in `docs/`. Skills should contain the package
contracts and procedures that change a coding agent's implementation decisions.

When a package contract changes, update the relevant skill and its examples.
Check the frontmatter, local reference links, and discovery links. Run any
changed executable example with local resources. Test the skill on a realistic
request when its workflow changes. Formatting checks alone do not prove that its
instructions are useful.

For a new skill, add matching discovery links in both directories. For example,
after creating `skills/langgraph-toolkit-example/SKILL.md`:

```sh
ln -s ../../skills/langgraph-toolkit-example .agents/skills/langgraph-toolkit-example
ln -s ../../skills/langgraph-toolkit-example .claude/skills/langgraph-toolkit-example
```

## Sources and examples

This layout uses the portable skill format and client-specific discovery
directories. The examples below informed the organization; the instructions in
this collection describe this toolkit.

- [Agent Skills specification](https://agentskills.io/specification): required
  frontmatter and optional resources.
- [OpenAI skill documentation](https://learn.chatgpt.com/docs/build-skills):
  Codex discovery and symbolic links.
- [Claude Code skills](https://code.claude.com/docs/en/skills): project skills
  and invocation.
- [LangChain skills](https://github.com/langchain-ai/langchain-skills):
  task-specific skills and conditional references.
- [Vercel agent skills](https://github.com/vercel-labs/agent-skills): a public
  `skills/` collection.
- [Anthropic skills](https://github.com/anthropics/skills): examples with
  references and executable helpers.
- [Skills CLI commands](https://github.com/vercel-labs/skills/blob/main/docs/commands.md):
  local and repository installation.
