"""Health endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from secondbrain.api.dependencies import get_health_service
from secondbrain.application.health import HealthService

router = APIRouter(tags=["health"])
HealthServiceDependency = Annotated[HealthService, Depends(get_health_service)]


@router.get("/health", summary="Check service health")
def health_check(service: HealthServiceDependency) -> dict[str, str]:
    """Report whether the API process is responding."""
    return service.check()
