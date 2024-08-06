import os
import sys
from http import HTTPStatus
from multiprocessing import Process
from typing import Callable, Any, Dict, List

import ray
import uvicorn
from fastapi import FastAPI
from httpx import Client
from pymoo.core.result import Result
from python_on_whales import DockerClient
from sqlmodel import SQLModel
from starlette.middleware.cors import CORSMiddleware

from autocode.container import ApplicationContainer
from autocode.datastore import OneDatastore
from autocode.model import OptimizationEvaluateRunResponse, OptimizationObjective, OptimizationInterpretation
from autocode.setting import ApplicationSetting
from autocode.use_case import OptimizationUseCase


class Optimization:
    def __init__(
            self,
            num_cpus: int,
            server_host: str,
            server_port: int,
            dashboard_port: int,
    ):
        super().__init__()
        if ray.is_initialized():
            ray.shutdown()

        ray.init(
            dashboard_host=server_host,
            num_cpus=num_cpus
        )
        print("Available resources: ", ray.available_resources())

        self.application_container: ApplicationContainer = ApplicationContainer()
        application_setting: ApplicationSetting = self.application_container.settings.application()
        application_setting.num_cpus = num_cpus
        application_setting.server_host = server_host
        application_setting.server_port = server_port
        application_setting.dashboard_port = dashboard_port
        one_datastore: OneDatastore = self.application_container.datastores.one()
        SQLModel.metadata.create_all(one_datastore.engine)

        self.app: FastAPI = FastAPI()
        self.app.add_middleware(
            middleware_class=CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )

        self.app.include_router(
            router=self.application_container.routers.api().router
        )

        self.server: Process = Process(
            target=self.run_server,
            daemon=True
        )

        while True:
            self.client: Client = Client(
                base_url=f"http://{server_host}:{server_port}/apis",
                timeout=None,
            )

            try:
                response = self.client.get("/healths")
            except Exception as e:
                if not self.server.is_alive():
                    self.server.start()
                continue

            if response.status_code != HTTPStatus.OK:
                raise ValueError(f"Error: {response.json()}")

            if response.status_code == HTTPStatus.OK:
                print("Server is healthy.")
                break

        self.dashboard: Process = Process(
            target=self.run_dashboard,
            daemon=True
        )
        self.dashboard.start()

    def run_dashboard(self):
        settings: ApplicationSetting = self.application_container.settings.application()
        os.system(
            f"python3 -m streamlit run {settings.absolute_path}/dashboard.py --server.port {settings.dashboard_port} > dashboard.log"
        )

    def run_server(self):
        sys.stdout = open("server.log", "w")
        sys.stderr = open("server.log", "w")
        uvicorn.run(
            app=self.app,
            host=self.application_container.settings.application().server_host,
            port=self.application_container.settings.application().server_port
        )

    def stop(self):
        self.server.terminate()
        ray.shutdown()

    def reset(self):
        optimization_use_case: OptimizationUseCase = self.application_container.use_cases.optimization()
        optimization_use_case.reset()

    def deploy(self, compose_files: List[str]) -> List[DockerClient]:
        optimization_use_case: OptimizationUseCase = self.application_container.use_cases.optimization()
        return optimization_use_case.deploy(
            compose_files=compose_files
        )

    def run(
            self,
            objectives: List[OptimizationObjective],
            num_inequality_constraints: int,
            num_equality_constraints: int,
            evaluator: Callable[[List[OptimizationEvaluateRunResponse]], Dict[str, Any]]
    ) -> Result:
        optimization_use_case: OptimizationUseCase = self.application_container.use_cases.optimization()
        return optimization_use_case.run(
            objectives=objectives,
            num_inequality_constraints=num_inequality_constraints,
            num_equality_constraints=num_equality_constraints,
            evaluator=evaluator,
        )

    def interpret(
            self,
            objectives: List[OptimizationObjective],
            result: Result,
            weights: List[float]
    ) -> OptimizationInterpretation:
        optimization_use_case: OptimizationUseCase = self.application_container.use_cases.optimization()
        return optimization_use_case.interpret(
            objectives=objectives,
            result=result,
            weights=weights
        )
