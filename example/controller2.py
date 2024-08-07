from autocode import OptimizationUseCase, ApplicationContainer, ApplicationSetting

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
    f_gateway: List[float] = []
    f_account: List[float] = []
    f_product: List[float] = []
    for input in inputs:
        if "gateway" in input.get_client().name:
            f_gateway = input.objectives
        elif "account" in input.get_client().name:
            f_account = input.objectives
        elif "product" in input.get_client().name:
            f_product = input.objectives

    output: Dict[str, Any] = {
        "F": f_gateway + f_account + f_product
    }

    return output


result = optimization_use_case.run(
    objectives=objectives,
    num_inequality_constraints=0,
    num_equality_constraints=0,
    evaluator=evaluator
)

print(result)
