import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from autocode import OptimizationUseCase, ApplicationContainer, ApplicationSetting

application_container: ApplicationContainer = ApplicationContainer()
application_setting: ApplicationSetting = application_container.settings.application()
application_setting.num_cpus = 2
application_setting.server_host = "0.0.0.0"
application_setting.server_port = 10000
application_setting.dashboard_port = 10001
optimization_use_case: OptimizationUseCase = application_container.use_cases.optimization()
# optimization_use_case.reset()

app: FastAPI = FastAPI()
app.add_middleware(
    middleware_class=CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(
    router=application_container.routers.api().router
)

uvicorn.run(
    app=app,
    host=application_setting.server_host,
    port=application_setting.server_port
)
