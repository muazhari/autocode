import asyncio
import uuid
from concurrent.futures import thread
from multiprocessing import Lock
from typing import Dict, List, Callable, Any, Coroutine, Optional, Literal

import dill
import numpy as np
from langchain.globals import get_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.utils import Output
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.core.algorithm import Algorithm
from pymoo.core.callback import Callback
from pymoo.core.mixed import MixedVariableSampling, MixedVariableMating, MixedVariableDuplicateElimination
from pymoo.core.plot import Plot
from pymoo.core.problem import ElementwiseProblem, RayParallelization
from pymoo.core.result import Result
from pymoo.core.variable import Variable, Binary, Choice, Integer, Real
from pymoo.decomposition.asf import ASF
from pymoo.optimize import minimize
from pymoo.visualization.pcp import PCP
from pymoo.visualization.scatter import Scatter
from python_on_whales import DockerClient
from python_on_whales.components.container.models import NetworkInspectResult
from ray.util.queue import Queue
from sqlalchemy import delete
from sqlmodel import Session, select

from autocode.datastore import OneDatastore
from autocode.gateway import EvaluationGateway
from autocode.model import OptimizationPrepareRequest, OptimizationChoice, OptimizationBinary, \
    OptimizationInteger, OptimizationReal, OptimizationClient, \
    OptimizationValue, OptimizationEvaluatePrepareRequest, OptimizationEvaluateRunResponse, OptimizationObjective, \
    OptimizationPrepareResponse, Cache, OptimizationValueFunction, CodeScoring, \
    ScoringState
from autocode.setting import ApplicationSetting


class OptimizationCallback(Callback):

    def __init__(
            self,
            one_datastore: OneDatastore,
    ):
        super().__init__()
        self.one_datastore = one_datastore

    def notify(self, algorithm: Algorithm):
        session: Session = self.one_datastore.get_session()
        session.begin()
        result: Result = algorithm.result()
        del result.algorithm
        del result.problem
        cache: Cache = Cache(
            key=f"results_{uuid.uuid4()}",
            value=dill.dumps(result)
        )
        session.add(cache)
        session.commit()
        session.close()


class OptimizationProblem(ElementwiseProblem):

    def __init__(
            self,
            application_setting: ApplicationSetting,
            optimization_gateway: EvaluationGateway,
            objectives: List[OptimizationObjective],
            variables: Dict[str, OptimizationBinary | OptimizationChoice | OptimizationInteger | OptimizationReal],
            clients: Dict[str, OptimizationClient],
            evaluator: Callable[[List[OptimizationEvaluateRunResponse]], Dict[str, Any]],
            *args,
            **kwargs
    ):
        self.application_setting = application_setting
        self.optimization_gateway = optimization_gateway
        self.objectives = objectives
        self.variables = variables
        self.clients = clients
        self.evaluator = evaluator

        self.client_to_worker_id_and_name: Dict[str, Dict[str, OptimizationClient]] = {}
        self.queue = Queue()
        for client in self.clients.values():
            if self.client_to_worker_id_and_name.get(client.worker_id, None) is None:
                self.queue.put(client.worker_id)
                self.client_to_worker_id_and_name[client.worker_id] = {}
            self.client_to_worker_id_and_name[client.worker_id][client.name] = client

        self.vars: Dict[str, Variable] = {}
        for variable_id, variable in variables.items():
            variable_type = type(variable)
            if variable_type == OptimizationBinary:
                self.vars[variable_id] = Binary()
            elif variable_type == OptimizationChoice:
                self.vars[variable_id] = Choice(options=list(variable.options.values()))
            elif variable_type == OptimizationInteger:
                self.vars[variable_id] = Integer(bounds=variable.bounds)
            elif variable_type == OptimizationReal:
                self.vars[variable_id] = Real(bounds=variable.bounds)
            else:
                raise ValueError(f"Variable type '{variable_type}' is not supported.")

        super().__init__(
            vars=self.vars,
            *args,
            **kwargs
        )

    def _evaluate(self, X, out, *args, **kwargs):
        worker_id: str = self.queue.get()

        client_to_variable_values: Dict[str, Dict[str, OptimizationValue]] = {}
        for variable_id, variable in self.variables.items():
            variable_value: Any = X[variable_id]
            if type(variable_value) is not OptimizationValue:
                variable_value: OptimizationValue = OptimizationValue(
                    data=variable_value
                )
            else:
                variable_value: OptimizationValue = variable_value

            if client_to_variable_values.get(variable.get_client_id(), None) is None:
                client_to_variable_values[variable.get_client_id()] = {}
            client_to_variable_values[variable.get_client_id()][variable_id] = variable_value

        prepare_futures: List[Coroutine] = []
        for client_id, variable_values in client_to_variable_values.items():
            variable_client: OptimizationClient = self.clients[client_id]
            worker_client: OptimizationClient = self.client_to_worker_id_and_name[worker_id][variable_client.name]

            prepare_request: OptimizationEvaluatePrepareRequest = OptimizationEvaluatePrepareRequest(
                variable_values=variable_values
            )
            prepare_response = self.optimization_gateway.evaluate_prepare(
                client=worker_client,
                request=prepare_request
            )
            prepare_futures.append(prepare_response)

        asyncio.get_event_loop().run_until_complete(asyncio.gather(*prepare_futures))

        run_futures: List[Coroutine] = []
        for worker_client in self.client_to_worker_id_and_name[worker_id].values():
            run_response = self.optimization_gateway.evaluate_run(
                client=worker_client,
            )
            run_futures.append(run_response)

        results: List[OptimizationEvaluateRunResponse] = asyncio.get_event_loop().run_until_complete(
            asyncio.gather(*run_futures)
        )

        self.queue.put(worker_id)

        evaluator_output: Dict[str, Any] = self.evaluator(results)
        evaluator_output.setdefault("F", [])
        evaluator_output.setdefault("G", [])
        evaluator_output.setdefault("H", [])

        if len(evaluator_output["F"]) != self.n_obj:
            raise ValueError(f"Number of objectives {len(evaluator_output['F'])} does not match {self.n_obj}.")
        if len(evaluator_output["G"]) != self.n_ieq_constr:
            raise ValueError(
                f"Number of inequality constraints {len(evaluator_output['G'])} does not match {self.n_ieq_constr}.")
        if len(evaluator_output["H"]) != self.n_eq_constr:
            raise ValueError(
                f"Number of equality constraints {len(evaluator_output['H'])} does not match {self.n_eq_constr}.")

        for index, objective in enumerate(self.objectives):
            if objective.type == "maximize":
                evaluator_output["F"][index] = -evaluator_output["F"][index]

        out.update(evaluator_output)


class LlmUseCase:
    def __init__(
            self,
            application_setting: ApplicationSetting
    ):
        self.application_setting = application_setting
        self.scoring_graph = self.get_scoring_graph()

    def get_scoring_graph(self):
        chat: ChatOpenAI = ChatOpenAI(
            model="gpt-4o-mini",
            max_tokens=4000,
            temperature=0,
            api_key=self.application_setting.OPENAI_API_KEY
        )

        parser: PydanticToolsParser = PydanticToolsParser(
            tools=[CodeScoring]
        )

        def node_scoring(state: ScoringState):
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template("""
                You are target group of our study. 
                The target group of our study are software quality analysts, researchers with a background in software quality, and software engineers that are involved with maintaining software. 
                Some participants have up to 15 years of experience in quality assessments. 
                In sum, 70 professionals participated. 
                First, we invited selected experts to participate in the study. 
                Second, we asked them to disseminate the study to interested and qualified colleagues. 
                The survey was also promoted in relevant networks. 
                The participants are affiliated with companies including Airbus, Audi, BMW, Boston Consulting Group, Celonis, cesdo Software Quality GmbH, CQSE GmbH, Facebook, fortiss, itestra GmbH, Kinexon GmbH, MaibornWolff GmbH, Munich Re, Oracle, and three universities. 
                However, 7 participants did not want to share their affiliation. 
                During the study, we followed a systematic approach to assess software maintainability. 
                The following steps were taken:
                """),
                HumanMessagePromptTemplate.from_template("""
                Comprehensively step-by-step analyze then score the existing code using a "CodeScoring" tool.
                Readability - this code is easy to read;
                Understandability - the semantic meaning of this code is clear;
                Complexity - this code is complex;
                Modularity  - this code should be broken into smaller pieces;
                Error Potentiality - this code is potentially error-prone;
                Overall maintainability - overall, this code is maintainable.
                The score scale from 0 (strongly disagree) to 100 (strongly agree).
                You must score the code in precision, example: 14.3, 42.456, 99.45, 78.58495, 3.141598, 0.579, etc.
                You must in context to the programming language.
                Ignore the function name, input-output parameters, and input-output types.
                <ProgrammingLanguage>
                {programming_language}
                </ProgrammingLanguage>
                <ExistingCode>
                {existing_code}
                </ExistingCode>
                """),
            ])
            chain: RunnableSerializable = (
                    prompt |
                    chat.bind_tools(
                        tools=parser.tools,
                        strict=True,
                        tool_choice="required"
                    ) |
                    parser
            )
            response: Output = chain.invoke({
                "programming_language": state["programming_language"],
                "existing_code": state["existing_code"],
            })
            state["score"] = response[0]

            return state

        graph = StateGraph(ScoringState)
        graph.set_entry_point(node_scoring.__name__)
        graph.add_node(node_scoring.__name__, node_scoring)
        graph.set_finish_point(node_scoring.__name__)
        compiled_graph = graph.compile()

        return compiled_graph

    async def function_scoring(self, language: str, function: OptimizationValueFunction) -> CodeScoring:
        state: ScoringState = await self.scoring_graph.ainvoke({
            "programming_language": language,
            "existing_code": function.string
        })

        return state["score"]


class OptimizationUseCase:
    def __init__(
            self,
            llm_use_case: LlmUseCase,
            evaluation_gateway: EvaluationGateway,
            one_datastore: OneDatastore,
            application_setting: ApplicationSetting
    ):
        self.llm_use_case = llm_use_case
        self.evaluation_gateway = evaluation_gateway
        self.one_datastore = one_datastore
        self.application_setting = application_setting
        self.client_locks: Dict[str, Lock] = {}

    def prepare(self, request: OptimizationPrepareRequest) -> OptimizationPrepareResponse:
        session: Session = self.one_datastore.get_session()
        session.begin()

        client_cache: Optional[Cache] = None
        client: Optional[OptimizationClient] = None
        client_caches: List[Cache] = list(
            session.exec(select(Cache).where(Cache.key.startswith("clients"))).all()
        )
        for client_cache in client_caches:
            current_client: OptimizationClient = dill.loads(client_cache.value)
            if current_client.host == request.host:
                client_cache = client_cache
                client = current_client
                client.variables = request.variables
                client.port = request.port
                client.is_ready = True
                break

        if client is None:
            raise ValueError(f"Client with host {request.host} is not found.")

        if self.client_locks.get(client.name, None) is None:
            self.client_locks[client.name] = Lock()
        self.client_locks[client.name].acquire()

        async def execution(function: OptimizationValueFunction):
            score: CodeScoring = await self.llm_use_case.function_scoring(
                language=request.language,
                function=function
            )
            function.understandability = score.understandability
            function.readability = score.readability
            function.complexity = score.complexity
            function.modularity = score.modularity
            function.error_potentiality = score.error_potentiality
            function.overall_maintainability = score.overall_maintainability

        list_scoring_futures: List[Coroutine] = []
        for variable_id, variable in client.variables.items():
            if variable.type == OptimizationChoice.__name__:
                for option_id, option in variable.options.items():
                    if option.type == OptimizationValueFunction.__name__:
                        function: OptimizationValueFunction = option.data
                        list_scoring_futures.append(execution(function))

        asyncio.set_event_loop(asyncio.new_event_loop())
        asyncio.get_event_loop().run_until_complete(asyncio.gather(*list_scoring_futures))

        for variable in client.variables.values():
            variable.set_client_id(client.id)

        client_cache.value = dill.dumps(client)
        session.add(client_cache)
        session.commit()
        session.close()
        self.client_locks[client.name].release()

        response: OptimizationPrepareResponse = OptimizationPrepareResponse(
            variables=client.variables,
        )

        return response

    def deploy(self, compose_files: List[str], num_workers: int) -> List[DockerClient]:
        self.reset(keys=["results", "objectives"])

        docker_clients: List[DockerClient] = []

        def execution(worker_id: str):
            session: Session = self.one_datastore.get_session()
            session.begin()
            docker_client_caches: List[Cache] = list(
                session.exec(select(Cache).where(Cache.key == f"docker_clients_{worker_id}")).all()
            )
            if len(docker_client_caches) == 1:
                docker_client_cache: Cache = docker_client_caches[0]
                docker_client: DockerClient = dill.loads(docker_client_cache.value)
            elif len(docker_client_caches) == 0:
                docker_client: DockerClient = DockerClient(
                    compose_files=compose_files,
                    compose_project_name=worker_id,
                )
                docker_client_cache: Cache = Cache(
                    key=f"docker_clients_{worker_id}",
                    value=dill.dumps(None)
                )
            else:
                raise ValueError(f"Number of docker client caches must be 0 or 1, but got {len(docker_client_caches)}.")

            docker_client.compose.stop()
            docker_client.compose.up(
                detach=True,
                build=True,
            )

            for container in docker_client.compose.ps():
                name: str = container.config.labels["com.docker.compose.service"]
                networks: List[NetworkInspectResult] = list(container.network_settings.networks.values())

                is_client_found: bool = False
                client_caches: List[Cache] = list(
                    session.exec(select(Cache).where(Cache.key.startswith("clients"))).all()
                )
                for client_cache in client_caches:
                    client: OptimizationClient = dill.loads(client_cache.value)
                    if client.name == name and client.worker_id == worker_id:
                        client.host = networks[0].ip_address
                        client.port = client.port
                        client.is_ready = False
                        client_cache.value = dill.dumps(client)
                        session.add(client_cache)
                        is_client_found = True

                if not is_client_found:
                    client: OptimizationClient = OptimizationClient(
                        variables={},
                        name=name,
                        host=networks[0].ip_address,
                        port=0,
                        worker_id=worker_id
                    )
                    client_cache: Cache = Cache(
                        key=f"clients_{client.id}",
                        value=dill.dumps(client)
                    )
                    session.add(client_cache)

            docker_clients.append(docker_client)
            docker_client_cache.value = dill.dumps(docker_client)
            session.add(docker_client_cache)
            session.commit()
            session.close()

        with thread.ThreadPoolExecutor() as executor:
            for i in range(num_workers):
                worker_id: str = f"autocode-worker-{i}"
                executor.submit(execution, worker_id)

        return docker_clients

    def reset(self, keys: Optional[List[Literal["docker_clients", "clients", "objectives", "results", "llms"]]] = None):
        session: Session = self.one_datastore.get_session()
        session.begin()
        if keys is None:
            keys = ["docker_clients", "clients", "objectives", "results", "llms"]

        if "docker_clients" in keys:
            docker_client_caches = list(
                session.exec(select(Cache).where(Cache.key.startswith("docker_clients"))).all()
            )

            def execution(docker_client_cache: Cache):
                docker_client: DockerClient = dill.loads(docker_client_cache.value)
                docker_client.compose.down()
                session.delete(docker_client_cache)

            with thread.ThreadPoolExecutor() as executor:
                for docker_client_cache in docker_client_caches:
                    executor.submit(execution, docker_client_cache)

        if "clients" in keys:
            self.client_locks = {}
            session.exec(delete(Cache).where(Cache.key.startswith("clients")))

        if "objectives" in keys:
            session.exec(delete(Cache).where(Cache.key == "objectives"))

        if "results" in keys:
            session.exec(delete(Cache).where(Cache.key.startswith("results")))

        if "llms" in keys:
            sqlite_cache: SQLiteCache = get_llm_cache()
            session.exec(delete(sqlite_cache.cache_schema))

        session.commit()
        session.close()

    def run(
            self,
            objectives: List[OptimizationObjective],
            num_inequality_constraints: int,
            num_equality_constraints: int,
            evaluator: Callable[[List[OptimizationEvaluateRunResponse]], Dict[str, Any]],
            job_resources: Dict[str, Any],
    ):
        self.reset(keys=["results", "objectives"])

        session: Session = self.one_datastore.get_session()
        session.begin()

        client_caches = list(session.exec(select(Cache).where(Cache.key.startswith("clients"))).all())
        objective_caches = list(session.exec(select(Cache).where(Cache.key == "objectives")).all())

        clients: Dict[str, OptimizationClient] = {}
        variables: Dict[str, OptimizationBinary | OptimizationChoice | OptimizationInteger | OptimizationReal] = {}

        for client_cache in client_caches:
            client: OptimizationClient = dill.loads(client_cache.value)

            if client.is_ready is False:
                raise ValueError(f"Client {client.id} is not ready.")

            variables.update(client.variables)
            clients[client.id] = client

        if len(objective_caches) == 0:
            objective_cache: Cache = Cache(
                key="objectives",
                value=dill.dumps(objectives)
            )
            session.add(objective_cache)
        else:
            raise ValueError(f"Number of objectives caches must be 0, but got {len(objective_caches)}.")

        if len(variables) == 0:
            raise ValueError("Number of variables must be greater than 0.")

        if len(clients) == 0:
            raise ValueError("Number of clients must be greater than 0.")

        session.commit()
        session.close()

        result: Result = self.minimize(
            objectives=objectives,
            num_inequality_constraints=num_inequality_constraints,
            num_equality_constraints=num_equality_constraints,
            variables=variables,
            evaluator=evaluator,
            clients=clients,
            job_resources=job_resources
        )

        if type(result.F) != np.ndarray or result.F.ndim == 1:
            result.F = np.array([result.F])

        del result.problem
        del result.algorithm

        return result

    def plot(self, result: Result, decision_index: int) -> Dict[str, Plot]:
        plot_scatter = Scatter()
        plot_scatter.add(result.F, color="blue")
        plot_scatter.add(result.F[decision_index], color="green")
        plot_scatter.show()

        plot_pcp = PCP()
        plot_pcp.add(result.F, color="blue")
        plot_pcp.add(result.F[decision_index], color="green")
        plot_pcp.show()

        plots: Dict[str, Plot] = {
            "scatter": plot_scatter,
            "pcp": plot_pcp
        }

        return plots

    def minimize(
            self,
            objectives: List[OptimizationObjective],
            num_inequality_constraints: int,
            num_equality_constraints: int,
            variables: Dict[str, OptimizationBinary | OptimizationChoice | OptimizationInteger | OptimizationReal],
            evaluator: Callable[[List[OptimizationEvaluateRunResponse]], Dict[str, Any]],
            clients: Dict[str, OptimizationClient],
            job_resources: Dict[str, Any],
    ) -> Result:
        runner: RayParallelization = RayParallelization(
            job_resources=job_resources
        )

        problem: OptimizationProblem = OptimizationProblem(
            application_setting=self.application_setting,
            optimization_gateway=self.evaluation_gateway,
            objectives=objectives,
            variables=variables,
            clients=clients,
            evaluator=evaluator,
            n_obj=len(objectives),
            n_ieq_constr=num_inequality_constraints,
            n_eq_constr=num_equality_constraints,
            elementwise_runner=runner
        )

        algorithm: AGEMOEA2 = AGEMOEA2(
            sampling=MixedVariableSampling(),
            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
            eliminate_duplicates=MixedVariableDuplicateElimination(),
        )

        callback: OptimizationCallback = OptimizationCallback(
            one_datastore=self.one_datastore
        )

        result: Result = minimize(
            problem=problem,
            algorithm=algorithm,
            callback=callback,
            seed=1,
            verbose=True
        )

        return result

    def get_decision_index(self, result: Result, weights: List[float]) -> int:
        weights = np.asarray(weights, dtype=np.float64)
        sum_weights = np.sum(weights)

        normalized_weights = weights / sum_weights if sum_weights != 0 else np.ones_like(weights) / len(weights)
        normalized_weights[normalized_weights == 0] += np.finfo(normalized_weights.dtype).eps

        ideal_point = np.min(result.F, axis=0)
        nadir_point = np.max(result.F, axis=0)
        scale = nadir_point - ideal_point
        scale[scale == 0] += np.finfo(scale.dtype).eps
        normalized_objectives = (result.F - ideal_point) / scale

        decomposition = ASF()
        mcdm = decomposition.do(normalized_objectives, 1 / normalized_weights)
        decision_index = np.argmin(mcdm)

        return decision_index
