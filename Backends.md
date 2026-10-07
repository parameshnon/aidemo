# Backends

## What a backend does

A backend is the layer through which an agent accesses files or other
workspace-backed data. In Deep Agents, a backend can provide filesystem
operations such as listing, reading, and writing. The backend determines
where those operations apply and may support virtual paths, storage, and
permissions.

The backend is an important capability boundary: the agent should not be able
to access the whole machine just because a task needs one project directory.

## FilesystemBackend examples

The repository uses `FilesystemBackend` in:

- `langraph\langchain-poc-deepagents.py`, rooted at the user-selected Git
  checkout.
- `langraph\langchain-deepAgents-backend.py`, rooted at
  `workspace\files`.
- `langchain-poc\langchain-poc-deepagent-skills.py`, rooted at the project so
  the skills directory can be loaded.

Example configuration:

```python
backend = FilesystemBackend(
    root_dir=project_root,
    virtual_mode=True,
)
```

The agent uses backend paths relative to the configured root rather than
arbitrary operating-system paths.

## Workspace-bound generation example

The architecture recommendation sample:

1. Checks that `workspace\files\brs.md` exists and is non-empty.
2. Creates a `FilesystemBackend` rooted specifically at `workspace\files`.
3. Reads `/brs.md` through the backend.
4. Passes the requirements as data to the agent.
5. Writes the generated Markdown to `/architecture_recommendation.md` using
   the same backend.

This confines the sample's file operations to the designated workspace
directory.

## Permissions and protected paths

The project assistant excludes direct shell and deletion tools, supplies a
fixed allow-listed test tool, and configures deny permissions for `.git`,
environment files, and common credential/key paths. The backend and permission
rules, not prompt wording alone, should enforce file boundaries.

## Backend design checklist

- Choose the narrowest root directory that supports the task.
- Resolve and validate roots before constructing an agent.
- Decide whether paths are virtual or direct and document that behavior.
- Deny reads and writes to secrets, VCS metadata, and generated or protected
  files where appropriate.
- Surface backend errors; do not silently substitute empty content or claim a
  failed write succeeded.
- Apply size limits and consider symlinks, path traversal, and concurrent
  modification.
- Treat documents read by the agent as untrusted input.

## Not a replacement for operating-system security

A backend limits the agent's intended file interface, but it does not replace
process isolation, least-privilege accounts, network controls, or safe handling
of tools that execute project code.
