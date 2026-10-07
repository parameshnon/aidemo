# Skills

## What a skill is

A skill is a reusable bundle of task-specific instructions and, optionally,
supporting files. It helps an agent follow a consistent procedure for a certain
kind of work without putting every instruction into the agent's global system
prompt.

Skills are instructions, not executable tools. A skill can tell an agent how to
write, review, or investigate; it does not itself grant filesystem or network
access. The agent still needs appropriate tools and permissions to perform
actions.

## Skill files

The example in
`langchain-poc\skills\concise-writing\SKILL.md` uses YAML frontmatter followed
by Markdown instructions:

```markdown
---
name: concise-writing
description: Use when the user asks to draft, rewrite, summarize, or polish written content.
---

# Concise Writing

When this skill applies:

1. Identify the audience and purpose.
2. Put the main point first.
3. Preserve the user's facts and intent.
```

The `name` identifies the skill and `description` helps the agent determine when
it is relevant. The body contains the workflow and constraints. Keep the
description concrete so the skill can be selected for suitable requests.

## Loading a skill with Deep Agents

`langchain-poc\langchain-poc-deepagent-skills.py` points the Deep Agent at the
project's skill directory:

```python
return create_deep_agent(
    model=model,
    backend=FilesystemBackend(root_dir=PROJECT_DIRECTORY, virtual_mode=True),
    skills=["/skills/"],
)
```

The backend provides the scoped filesystem context and the `skills` argument
identifies the directory to load. The sample checks that the skills directory
exists before constructing the agent.

## Good skill design

- Make a skill focused on one workflow or domain.
- State when the skill applies and what a good result looks like.
- Use short, actionable instructions instead of duplicating the entire system
  prompt.
- Tell the agent what not to assume or invent.
- Include examples or reference files only when they materially improve the
  workflow.
- Review a skill as code: incorrect instructions can cause repeatable incorrect
  behavior.

## Security and limits

Instructions do not enforce access controls. A skill that says “do not read a
secret” cannot replace backend permissions that deny access to secret files.
Use tools, backend boundaries, and permission checks to enforce what the agent
can actually do.
