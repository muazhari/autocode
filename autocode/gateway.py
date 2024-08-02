from http import HTTPStatus

from httpx import AsyncClient, AsyncHTTPTransport

from autocode.model import OptimizationEvaluatePrepareRequest, OptimizationClient, \
    OptimizationEvaluateRunRequest, OptimizationEvaluateRunResponse


class EvaluationGateway:
    def __init__(self):
        pass

    async def evaluate_prepare(self, client: OptimizationClient, request: OptimizationEvaluatePrepareRequest):
        client: AsyncClient = AsyncClient(
            base_url=f"http://{client.host}:{client.port}/apis",
            transport=AsyncHTTPTransport(retries=30)
        )
        response = await client.post(
            url="/optimizations/evaluates/prepares",
            json=request.model_dump(mode="json"),
            timeout=None
        )

        if response.status_code != HTTPStatus.OK:
            raise ValueError(f"Error: {response.json()}")

    async def evaluate_run(self, client: OptimizationClient, request: OptimizationEvaluateRunRequest):
        client: AsyncClient = AsyncClient(
            base_url=f"http://{client.host}:{client.port}/apis",
            transport=AsyncHTTPTransport(retries=30)
        )
        response = await client.post(
            url="/optimizations/evaluates/runs",
            json=request.model_dump(mode="json"),
            timeout=None
        )

        if response.status_code != HTTPStatus.OK:
            raise ValueError(f"Error: {response.json()}")

        data: OptimizationEvaluateRunResponse = OptimizationEvaluateRunResponse(**response.json())

        return data
