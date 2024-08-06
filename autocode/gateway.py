from http import HTTPStatus

from httpx import AsyncClient, AsyncHTTPTransport

from autocode.model import OptimizationEvaluatePrepareRequest, OptimizationClient, OptimizationEvaluateRunResponse


class EvaluationGateway:
    def __init__(self):
        pass

    async def evaluate_prepare(self, client: OptimizationClient, request: OptimizationEvaluatePrepareRequest):
        async_client: AsyncClient = AsyncClient(
            base_url=f"http://{client.host}:{client.port}/apis",
            transport=AsyncHTTPTransport(retries=30)
        )
        response = await async_client.post(
            url="/optimizations/evaluates/prepares",
            json=request.model_dump(mode="json"),
            timeout=None
        )

        if response.status_code != HTTPStatus.OK:
            raise ValueError(f"Error: {response.json()}")

    async def evaluate_run(self, client: OptimizationClient):
        async_client: AsyncClient = AsyncClient(
            base_url=f"http://{client.host}:{client.port}/apis",
            transport=AsyncHTTPTransport(retries=30)
        )
        response = await async_client.get(
            url="/optimizations/evaluates/runs",
            timeout=None
        )

        if response.status_code != HTTPStatus.OK:
            raise ValueError(f"Error: {response.json()}")

        data: OptimizationEvaluateRunResponse = OptimizationEvaluateRunResponse(**response.json())
        data.set_client(client)

        return data
