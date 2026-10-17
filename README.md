# SecondBrain AI

A personal knowledge management application. This repository currently contains the engineering foundation for the 10-week project.

## Local setup

Requires Python 3.11 through 3.14.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
uvicorn secondbrain.main:app --reload --app-dir src
```

The API is available at `http://127.0.0.1:8000`; the health endpoint is `GET /health` and the OpenAPI UI is `/docs`.

Configuration is loaded from environment variables with the `SB_` prefix. Local `.env` values are optional; defaults are defined in `secondbrain.config.settings.Settings`.

## Quality checks

Run these from the repository root:

```bash
pytest
ruff check .
ruff format --check .
mypy src
```

## Project docs

- [Requirements](docs/requirements.md)
- [Initial architecture](docs/architecture.md)
