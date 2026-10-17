# Initial architecture

## Current request flow

`HTTP request → FastAPI route → application service → response`

The health route obtains `HealthService` through FastAPI dependency injection. This keeps request handling separate from application behavior and allows tests or future implementations to override dependencies cleanly.

## Code boundaries

- `api/`: HTTP routes and dependency providers.
- `application/`: use-case behavior independent of HTTP and infrastructure.
- `config/`: validated runtime settings sourced from `SB_` environment variables and an optional local `.env` file.
- `infrastructure/`: reserved for concrete external adapters when needed; intentionally empty for this session.
- `main.py`: application factory and ASGI entry point.

## Future direction

New capabilities should be introduced behind application-level interfaces only when a real integration needs them. Local and AWS-backed implementations can then be selected in infrastructure wiring without coupling route handlers to storage or cloud SDKs. No persistence, AI provider, or cloud assumptions are made in this foundation.
