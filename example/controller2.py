import os

from autocode import OptimizationUseCase, ApplicationContainer, ApplicationSetting

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "autocode"

application_container: ApplicationContainer = ApplicationContainer()
application_setting: ApplicationSetting = application_container.settings.application()
application_setting.num_cpus = 2
application_setting.server_host = "0.0.0.0"
application_setting.server_port = 10000
application_setting.dashboard_port = 10001
optimization_use_case: OptimizationUseCase = application_container.use_cases.optimization()
# optimization_use_case.reset()

from autocode import OptimizationObjective
from typing import Dict, Any, List
from autocode import OptimizationEvaluateRunResponse

objectives: List[OptimizationObjective] = [
    OptimizationObjective(
        type="minimize",
    ),
    OptimizationObjective(
        type="maximize",
    ),
    OptimizationObjective(
        type="maximize",
    ),
    OptimizationObjective(
        type="minimize",
    ),
    OptimizationObjective(
        type="maximize",
    ),
    OptimizationObjective(
        type="minimize",
    ),
    OptimizationObjective(
        type="maximize",
    ),

    OptimizationObjective(
        type="minimize",
    ),
    OptimizationObjective(
        type="maximize",
    ),
    OptimizationObjective(
        type="maximize",
    ),
    OptimizationObjective(
        type="minimize",
    ),
    OptimizationObjective(
        type="maximize",
    ),
    OptimizationObjective(
        type="minimize",
    ),
    OptimizationObjective(
        type="maximize",
    ),
]


def evaluator(inputs: List[OptimizationEvaluateRunResponse]) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "F": inputs[0].objectives + inputs[1].objectives,
    }

    return output


result = optimization_use_case.run(
    objectives=objectives,
    num_inequality_constraints=0,
    num_equality_constraints=0,
    evaluator=evaluator
)

print(result)
