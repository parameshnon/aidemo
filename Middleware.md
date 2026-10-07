# Middleware

## What middleware is

Middleware wraps agent execution at defined lifecycle points. It is useful for
cross-cutting behavior such as validation, authorization, logging, retries,
policy enforcement, and shaping model inputs or outputs. Middleware provides
one place to apply these concerns consistently around model calls.

Middleware is distinct from a tool: a tool performs a task at the model's
request, while middleware runs as part of the agent lifecycle.

## Hooks in the examples

This repository uses LangChain agent middleware decorators:

- `@before_model` runs before the model call.
- `@after_model` runs after the model returns.

`langraph\langgrapn-middleware.py` demonstrates input validation,
fail-closed token authorization, and request/response logging. The simpler
`langraph\Langchain-poc-middleware.py` demonstrates input length validation,
logging, a word-count tool, and structured output.

## Input validation

The validation hook inspects the most recent user message and rejects a missing,
empty, non-text, or overlong request before it reaches the model. Validate
structured values at the boundary where they enter the application, and return
an explicit error rather than a success-shaped fallback.

## Authorization

The authorization example retrieves the expected secret from
`CONCIERGE_ACCESS_TOKEN`, gets the caller's token from runtime context, and
compares them with `secrets.compare_digest`. It fails closed if the server
secret is unset or the supplied token is absent/incorrect.

The access token is passed as runtime context rather than placed in the prompt.
The example is educational; production authentication should use the
application's identity provider and a suitable credential/session design.

## Logging

The examples log counts and lifecycle events instead of prompt bodies or
tokens. This reduces accidental exposure, but metadata can still be sensitive.
Choose log retention, access, and redaction policies deliberately.

## Structured output

Middleware can be combined with a Pydantic response model. The examples define
response fields such as `answer`, `category`, and `key_points`, then validate
that `structured_response` has the expected type before serializing it.
Structured output improves shape reliability; it does not guarantee that the
content is factually correct.

## Middleware checklist

- Put deterministic checks before expensive model calls.
- Fail closed for authorization.
- Do not log secrets or full user content by default.
- Keep hooks small and make their ordering understandable.
- Raise or return errors using the application's standard error handling.
- Add tests for allowed and denied requests and for boundary input values.
