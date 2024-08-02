from fastapi import APIRouter

from autocode.controller import OptimizationController, HealthController


class ApiRouter:
    def __init__(
            self,
            optimization_controller: OptimizationController,
            health_controller: HealthController,
    ):
        self.optimization_controller = optimization_controller
        self.health_controller = health_controller
        self.router = APIRouter(
            prefix="/apis",
            tags=["api"]
        )
        self.router.include_router(optimization_controller.router)
        self.router.include_router(health_controller.router)
