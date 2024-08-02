import asyncio
import copy
import re
import uuid
from typing import Dict, List, Callable, Any, Coroutine

import dill
import numpy as np
import ray
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    AIMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.algorithm import Algorithm
from pymoo.core.callback import Callback
from pymoo.core.mixed import MixedVariableSampling, MixedVariableMating, MixedVariableDuplicateElimination
from pymoo.core.plot import Plot
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.result import Result
from pymoo.core.variable import Variable, Binary, Choice, Integer, Real
from pymoo.decomposition.asf import ASF
from pymoo.optimize import minimize
from pymoo.visualization.pcp import PCP
from pymoo.visualization.scatter import Scatter
from ray.util.queue import Queue
from sqlalchemy import delete
from sqlmodel import Session, SQLModel, select

from autocode.datastore import OneDatastore
from autocode.gateway import EvaluationGateway
from autocode.model import OptimizationPrepareRequest, OptimizationChoice, OptimizationBinary, \
    OptimizationInteger, OptimizationReal, OptimizationVariable, OptimizationClient, OptimizationEvaluateRunRequest, \
    OptimizationValue, OptimizationEvaluatePrepareRequest, OptimizationEvaluateRunResponse, OptimizationObjective, \
    OptimizationPrepareResponse, Cache, OptimizationValueFunction, CodeScoring, \
    ScoringState, CodeVariation, VariationState
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
        self.queue = Queue()
        for i in range(self.application_setting.num_cpus):
            self.queue.put(str(i))
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
            n_var=len(self.vars),
            *args,
            **kwargs
        )

    def _evaluate(self, X, out, *args, **kwargs):
        worker_id: str = self.queue.get()

        client_to_variable_values: Dict[str, Dict[str, OptimizationValue]] = {}
        for variable_id, variable in self.variables.items():
            value: Any = X[variable_id]
            if type(value) is not OptimizationValue:
                value: OptimizationValue = OptimizationValue(
                    data=value
                )
            else:
                value: OptimizationValue = value

            if client_to_variable_values.get(variable.get_client_id(), None) is None:
                client_to_variable_values[variable.get_client_id()] = {}
            client_to_variable_values[variable.get_client_id()][variable_id] = value

        prepare_futures: List[Coroutine] = []
        for client_id, variable_values in client_to_variable_values.items():
            prepare_request: OptimizationEvaluatePrepareRequest = OptimizationEvaluatePrepareRequest(
                worker_id=worker_id,
                variable_values=variable_values
            )
            client: OptimizationClient = self.clients[client_id]
            prepare_response = self.optimization_gateway.evaluate_prepare(
                client=client,
                request=prepare_request
            )
            prepare_futures.append(prepare_response)

        asyncio.get_event_loop().run_until_complete(asyncio.gather(*prepare_futures))

        run_futures: List[Coroutine] = []
        for client in self.clients.values():
            run_request: OptimizationEvaluateRunRequest = OptimizationEvaluateRunRequest(
                worker_id=worker_id
            )
            run_response = self.optimization_gateway.evaluate_run(
                client=client,
                request=run_request
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
        self.variation_graph = self.get_variation_graph()

    def get_scoring_graph(self):
        chat = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=self.application_setting.openai_api_key
        )

        tools = [CodeScoring]
        parser = PydanticToolsParser(tools=tools)

        def node_scoring_analyze(state: ScoringState):
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template("""
                    You are target group of our study. The target group of our study are software quality analysts, researchers with a background in software quality, and software engineers that are involved with maintaining software. Some participants have up to 15 years of experience in quality assessments. In sum, 70 professionals participated. First, we invited selected experts to participate in the study. Second, we asked them to disseminate the study to interested and qualified colleagues. The survey was also promoted in relevant net- works. The participants are affiliated with companies including Airbus, Audi, BMW, Boston Consulting Group, Celonis, cesdo Software Quality GmbH, CQSE GmbH, Facebook, fortiss, itestra GmbH, Kinexon GmbH, MaibornWolff GmbH, Munich Re, Oracle, and three universities. However, 7 participants did not want to share their affiliation. During the study, we followed a systematic approach to assess software maintainability. The following steps were taken:
                    """),
                HumanMessagePromptTemplate.from_template("""
                    Analyze readability, understandability, complexity, modularity, and overall maintainability metrics of the following code.
                    You must use step by step comprehensive reasoning to explain your analysis.
                    <code>
                    {code}
                    </code>
                    """),
            ])
            chain = prompt | chat
            response = chain.invoke({
                "code": state["code"],
            })
            state["analysis"] = response.content
            return state

        def node_scoring(state: ScoringState):
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template("""
                        You are target group of our study. The target group of our study are software quality analysts, researchers with a background in software quality, and software engineers that are involved with maintaining software. Some participants have up to 15 years of experience in quality assessments. In sum, 70 professionals participated. First, we invited selected experts to participate in the study. Second, we asked them to disseminate the study to interested and qualified colleagues. The survey was also promoted in relevant networks. The participants are affiliated with companies including Airbus, Audi, BMW, Boston Consulting Group, Celonis, cesdo Software Quality GmbH, CQSE GmbH, Facebook, fortiss, itestra GmbH, Kinexon GmbH, MaibornWolff GmbH, Munich Re, Oracle, and three universities. However, 7 participants did not want to share their affiliation. During the study, we followed a systematic approach to assess software maintainability. The following steps were taken:
                        """),
                HumanMessagePromptTemplate.from_template("""
                        Analyze error potentiality, readability, understandability, complexity, modularity, and overall maintainability metrics of the following code.
                        You must use step by step comprehensive reasoning to explain your analysis.
                        <code>
                        {code}
                        </code>
                        """),
                AIMessagePromptTemplate.from_template("""
                        {analysis}
                        """),
                HumanMessagePromptTemplate.from_template("""
                        Based on your analysis and the provided code, score the code based on the following criteria:
                        Error potentiality - this code is potentially error-prone;
                        Readability - this code is easy to read; 
                        Understandability - the semantic meaning of this code is clear; 
                        Complexity - this code is complex; 
                        Modularity  - this code should be broken into smaller pieces; 
                        Overall maintainability - overall, this code is maintainable. 
                        The score scale from 1 (strongly agree) to 100 (strongly disagree).
                        You must score in precision, i.e. 14, 47.456, 75, 58.58495, 3.141598, etc.
                        """),
            ])
            chain = prompt | chat.bind_tools(tools=tools) | parser
            response = chain.invoke({
                "code": state["code"],
                "analysis": state["analysis"]
            })
            state["score"] = response
            return state

        graph = StateGraph(ScoringState)
        graph.set_entry_point(node_scoring_analyze.__name__)
        graph.add_node(node_scoring_analyze.__name__, node_scoring_analyze)
        graph.add_node(node_scoring.__name__, node_scoring)
        graph.add_edge(node_scoring_analyze.__name__, node_scoring.__name__)
        graph.set_finish_point(node_scoring.__name__)
        compiled_graph = graph.compile()

        return compiled_graph

    def get_variation_graph(self):
        chat = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=self.application_setting.openai_api_key
        )

        tools = [CodeVariation]
        parser = PydanticToolsParser(tools=tools)

        def node_variation_analyze(state: VariationState):
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template("""
            You are target group of our study. The target group of our study are software quality analysts, researchers with a background in software quality, and software engineers that are involved with maintaining software. Some participants have up to 15 years of experience in quality assessments. In sum, 70 professionals participated. First, we invited selected experts to participate in the study. Second, we asked them to disseminate the study to interested and qualified colleagues. The survey was also promoted in relevant net- works. The participants are affiliated with companies including Airbus, Audi, BMW, Boston Consulting Group, Celonis, cesdo Software Quality GmbH, CQSE GmbH, Facebook, fortiss, itestra GmbH, Kinexon GmbH, MaibornWolff GmbH, Munich Re, Oracle, and three universities. However, 7 participants did not want to share their affiliation. During the study, we followed a systematic approach to assess software maintainability. The following steps were taken:
            """),
                HumanMessagePromptTemplate.from_template("""
            Analyze the following code to propose many possible variations.
            <code>
            {code}
            </code>
            """),
            ])
            chain = prompt | chat
            response = chain.invoke({
                "code": state["code"],
            })
            state["analysis"] = response.content
            return state

        def node_variation(state: VariationState):
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template("""
                You are target group of our study. The target group of our study are software quality analysts, researchers with a background in software quality, and software engineers that are involved with maintaining software. Some participants have up to 15 years of experience in quality assessments. In sum, 70 professionals participated. First, we invited selected experts to participate in the study. Second, we asked them to disseminate the study to interested and qualified colleagues. The survey was also promoted in relevant networks. The participants are affiliated with companies including Airbus, Audi, BMW, Boston Consulting Group, Celonis, cesdo Software Quality GmbH, CQSE GmbH, Facebook, fortiss, itestra GmbH, Kinexon GmbH, MaibornWolff GmbH, Munich Re, Oracle, and three universities. However, 7 participants did not want to share their affiliation. During the study, we followed a systematic approach to assess software maintainability. The following steps were taken:
                """),
                HumanMessagePromptTemplate.from_template("""
                Analyze the following code to propose many possible variations.
                <code>
                {code}
                </code>
                """),
                AIMessagePromptTemplate.from_template("""
                {analysis}
                """),
                HumanMessagePromptTemplate.from_template("""
                Generate 1 variations of the following code using "CodeVariation" tools.
                Ensure do not using libraries.
                Do not change the function parameters.
                """),
            ])
            chain = prompt | chat.bind_tools(tools=tools) | parser
            response = chain.invoke({
                "code": state["code"],
                "analysis": state["analysis"],
            })
            state["variation"] = response
            return state

        graph = StateGraph(VariationState)
        graph.set_entry_point(node_variation_analyze.__name__)
        graph.add_node(node_variation_analyze.__name__, node_variation_analyze)
        graph.add_node(node_variation.__name__, node_variation)
        graph.add_edge(node_variation_analyze.__name__, node_variation.__name__)
        graph.set_finish_point(node_variation.__name__)
        compiled_graph = graph.compile()

        return compiled_graph

    async def function_scoring(self, function: OptimizationValueFunction) -> List[CodeScoring]:
        state: ScoringState = await self.scoring_graph.ainvoke({
            "code": function.string
        })

        return state["score"]

    async def generate_function_variation(self, function: OptimizationValueFunction) -> List[CodeVariation]:
        state: VariationState = await self.variation_graph.ainvoke({
            "code": function.string,
        })

        return state["variation"]


class OptimizationProblemRunner:
    def __init__(self):
        pass

    def __call__(self, f, X):
        runnable = ray.remote(f.__call__.__func__)
        futures = [runnable.remote(f, x) for x in X]
        return ray.get(futures)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state


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

    def prepare(self, request: OptimizationPrepareRequest) -> OptimizationPrepareResponse:
        client: OptimizationClient = OptimizationClient(
            variables=request.variables,
            host=request.host,
            port=request.port,
            name=request.name
        )

        list_variation_futures: List[Coroutine] = []
        list_variables: List[OptimizationVariable] = []
        list_functions: List[OptimizationValueFunction] = []

        for variable_id, variable in client.variables.items():
            variable.set_client_id(client.id)

            if type(variable) is OptimizationChoice:
                for option_id, option in variable.options.items():
                    if option.type == "OptimizationValueFunction":
                        function: OptimizationValueFunction = option.data
                        future_variation: Coroutine = self.llm_use_case.generate_function_variation(function)
                        list_variation_futures.append(future_variation)
                        list_variables.append(variable)
                        list_functions.append(function)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        list_variations = asyncio.get_event_loop().run_until_complete(asyncio.gather(*list_variation_futures))
        for variable, variations, function in zip(list_variables, list_variations, list_functions):
            for variation in variations:
                new_option_id: str = str(uuid.uuid4())
                copied_function: OptimizationValueFunction = copy.deepcopy(function)
                new_function_name: str = "variation_" + uuid.uuid4().hex
                split_function_name = copied_function.name.split(".")
                split_function_name[-1] = new_function_name
                copied_function.name = ".".join(split_function_name)
                copied_function.string = re.sub(
                    pattern=r"func (.+?)\(",
                    repl="func " + new_function_name + "(",
                    string=variation.variation
                )
                variable.options[new_option_id] = OptimizationValue(
                    id=new_option_id,
                    data=copied_function
                )

        list_functions: List[OptimizationValueFunction] = []
        list_scoring_futures: List[Coroutine] = []
        for variable_id, variable in client.variables.items():
            if type(variable) is OptimizationChoice:
                for option_id, option in variable.options.items():
                    if option.type == "OptimizationValueFunction":
                        future_scoring: Coroutine = self.llm_use_case.function_scoring(option.data)
                        list_scoring_futures.append(future_scoring)
                        list_functions.append(option.data)

        list_scores = asyncio.get_event_loop().run_until_complete(asyncio.gather(*list_scoring_futures))
        for function, score in zip(list_functions, list_scores):
            function.understandability = score[0].understandability
            function.error_potentiality = score[0].error_potentiality
            function.readability = score[0].readability
            function.complexity = score[0].complexity
            function.modularity = score[0].modularity
            function.overall_maintainability = score[0].overall_maintainability

        session: Session = self.one_datastore.get_session()
        client_cache: Cache = Cache(
            key=f"clients_{client.id}",
            value=dill.dumps(client)
        )
        session.add(client_cache)
        session.commit()
        session.close()

        response = OptimizationPrepareResponse(
            variables=client.variables,
            num_workers=self.application_setting.num_cpus
        )

        return response

    def run_deployment(self):
        pass


    def reset(self):
        SQLModel.metadata.drop_all(self.one_datastore.engine)
        SQLModel.metadata.create_all(self.one_datastore.engine)

    def run(
            self,
            objectives: List[OptimizationObjective],
            num_inequality_constraints: int,
            num_equality_constraints: int,
            evaluator: Callable[[List[OptimizationEvaluateRunResponse]], Dict[str, Any]],
    ):
        variables: Dict[str, OptimizationBinary | OptimizationChoice | OptimizationInteger | OptimizationReal] = {}
        session: Session = self.one_datastore.get_session()
        session.exec(delete(Cache).where(Cache.key.startswith("results")))
        session.exec(delete(Cache).where(Cache.key == "objectives"))

        client_caches = list(session.exec(select(Cache).where(Cache.key.startswith("clients"))).all())
        objective_caches = list(session.exec(select(Cache).where(Cache.key == "objectives")).all())

        clients: Dict[str, OptimizationClient] = {}
        for client_cache in client_caches:
            client: OptimizationClient = dill.loads(client_cache.value)
            clients[client.id] = client
            variables.update(client.variables)

        if len(objective_caches) == 0:
            objective_cache: Cache = Cache(
                key="objectives",
                value=dill.dumps(objectives)
            )
            session.add(objective_cache)
        else:
            raise ValueError(f"Number of objectives caches must be 0, but got {len(objective_caches)}.")

        session.commit()
        session.close()

        result: Result = self.minimize(
            objectives=objectives,
            num_inequality_constraints=num_inequality_constraints,
            num_equality_constraints=num_equality_constraints,
            variables=variables,
            evaluator=evaluator,
            clients=clients
        )

        if type(result.F) != np.ndarray or result.F.ndim == 1:
            result.F = np.array([result.F])

        del result.problem
        del result.algorithm

        return result

    def plot(self, result: Result, decision_index: int) -> List[Plot]:
        plot_0 = Scatter()
        plot_0.add(result.F, color="blue")
        plot_0.add(result.F[decision_index], color="green")
        plot_0.show()

        plot_1 = PCP()
        plot_1.add(result.F, color="blue")
        plot_1.add(result.F[decision_index], color="green")
        plot_1.show()

        return [plot_0, plot_1]

    def minimize(
            self,
            objectives: List[OptimizationObjective],
            num_inequality_constraints: int,
            num_equality_constraints: int,
            variables: Dict[str, OptimizationBinary | OptimizationChoice | OptimizationInteger | OptimizationReal],
            evaluator: Callable[[List[OptimizationEvaluateRunResponse]], Dict[str, Any]],
            clients: Dict[str, OptimizationClient],
    ) -> Result:
        runner: OptimizationProblemRunner = OptimizationProblemRunner()

        problem = OptimizationProblem(
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

        algorithm: SMSEMOA = SMSEMOA(
            sampling=MixedVariableSampling(),
            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
            eliminate_duplicates=MixedVariableDuplicateElimination(),
        )

        callback: OptimizationCallback = OptimizationCallback(
            one_datastore=self.one_datastore
        )

        result: Result = minimize(
            problem,
            algorithm,
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
