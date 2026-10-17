"""Application-level health behavior."""


class HealthService:
    """Provides application health state without infrastructure dependencies."""

    def check(self) -> dict[str, str]:
        """Return the current service status."""
        return {"status": "ok"}
