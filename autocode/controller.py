from fastapi import APIRouter, Body

from autocode.model import OptimizationPrepareRequest, OptimizationPrepareResponse
from autocode.use_case import OptimizationUseCase


class OptimizationController:
    def __init__(
            self,
            optimization_use_case: OptimizationUseCase,
    ):
        self.optimization_use_case = optimization_use_case
        self.router: APIRouter = APIRouter(
            prefix="/optimizations",
            tags=["optimizations"]
        )
        self.router.add_api_route(
            path="/prepares",
            endpoint=self.prepare,
            methods=["POST"]
        )

    def prepare(self, request: OptimizationPrepareRequest = Body()) -> OptimizationPrepareResponse:
        response: OptimizationPrepareResponse = self.optimization_use_case.prepare(
            request=request
        )

        return response


class HealthController:
    def __init__(
            self,
    ):
        self.router: APIRouter = APIRouter()
        self.router.add_api_route(
            path="/healths",
            endpoint=self.health_check,
            methods=["GET"]
        )

    def health_check(self):
        pass