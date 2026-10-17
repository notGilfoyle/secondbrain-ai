"""FastAPI dependency providers."""

from secondbrain.application.health import HealthService


def get_health_service() -> HealthService:
    """Construct the health application service."""
    return HealthService()
