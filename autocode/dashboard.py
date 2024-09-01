import time
from math import ceil
from typing import List, Dict, Any, Set

import dill
import numpy as np
import pandas as pd
import streamlit as st
from pymoo.core.plot import Plot
from pymoo.core.result import Result
from sqlmodel import Session, select

from autocode import OptimizationUseCase
from autocode.container import ApplicationContainer
from autocode.datastore import OneDatastore
from autocode.model import Cache, OptimizationObjective, OptimizationVariable, OptimizationClient, OptimizationValue, \
    OptimizationBinary, OptimizationChoice, OptimizationInteger, OptimizationReal, OptimizationValueFunction

container: ApplicationContainer = ApplicationContainer()
one_datastore: OneDatastore = container.datastores.one()
optimization_use_case: OptimizationUseCase = container.use_cases.optimization()
objective_caches: List[Cache] = []
objectives: List[OptimizationObjective] = []
variables: Dict[str, OptimizationBinary | OptimizationReal | OptimizationInteger | OptimizationChoice] = {}
client_caches: List[Cache] = []
clients: Dict[str, OptimizationClient] = {}
weights: List[float] = []

session: Session = one_datastore.get_session()

st.session_state.setdefault("old_result_caches", set())

clear_cache = st.button("Clear cache")
if clear_cache:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.clear()

st.subheader("Clients")
client_df_placeholder = st.empty()

while True:
    try:
        objective_caches = list(session.exec(select(Cache).where(Cache.key == "objectives")).all())
        client_caches = list(session.exec(select(Cache).where(Cache.key.startswith("clients"))).all())
    except Exception as e:
        print(e)
        time.sleep(0.01)
        continue

    clients = {}
    variables = {}
    client_df_list: List[Dict[str, Any]] = []
    for client_cache in client_caches:
        client: OptimizationClient = dill.loads(client_cache.value)
        variables.update(client.variables)
        clients[client.id] = client
        client_json: Dict[str, Any] = client.model_dump(mode="json")
        client_json["port"] = str(client.port)
        client_df_list.append(client_json)

    client_df: pd.DataFrame = pd.DataFrame(client_df_list)

    with client_df_placeholder.container():
        st.dataframe(client_df, height=500)

    if len(objective_caches) > 0:
        break

    session.close()
    time.sleep(0.01)

if len(objective_caches) == 0 and len(client_caches) == 0:
    st.write("Waiting for preparation data.")
elif len(objective_caches) == 1 and len(client_caches) >= 1:
    objectives = dill.loads(objective_caches[0].value)
    for index, objective in enumerate(objectives):
        st.subheader(f"Objective {index + 1}")
        st.radio(
            label="Type",
            options=[objective.type],
            index=0,
            horizontal=True,
            key=f"type_{index}"
        )
        weight: float = st.slider(
            label=f"Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            key=f"weight_{index}"
        )
        weights.append(weight)
else:
    st.error("Preparation data cache is not valid.")

plot_placeholder = st.empty()

while True:
    try:
        query = select(Cache).where(Cache.key.startswith("results"))
        result_caches: Set[Cache] = set(session.exec(query).all())
        diff_result_caches = result_caches - st.session_state["old_result_caches"]
    except Exception as e:
        print(e)
        time.sleep(0.01)
        continue

    if len(diff_result_caches) == 0:
        to_display_indexes = []
    else:
        n: int = 10
        to_display_indexes = list(range(len(diff_result_caches)))[::int(ceil(len(diff_result_caches) / n))]

    for index, cache in enumerate(diff_result_caches):
        if index not in to_display_indexes:
            st.session_state["old_result_caches"].add(cache)
            continue

        result: Result = dill.loads(cache.value)

        decision_index: int = optimization_use_case.get_decision_index(
            result=result,
            weights=weights
        )
        for index, objective in enumerate(objectives):
            if objective.type == "maximize":
                result.F[:, index] = -result.F[:, index]

        plots: Dict[str, Plot] = optimization_use_case.plot(
            result=result,
            decision_index=decision_index
        )

        list_dict_x: List[List[Dict[str, Any]]] = []
        list_dict_f: List[Dict[str, Any]] = []
        for index, (x, f) in enumerate(zip(result.X, result.F)):
            list_transformed_x: List[Dict[str, Any]] = []
            for variable_id, variable_value in x.items():
                variable: OptimizationVariable = variables[variable_id]
                client: OptimizationClient = clients[variable.get_client_id()]
                if type(variable_value) is not OptimizationValue:
                    value: OptimizationValue = OptimizationValue(
                        data=variable_value
                    )
                else:
                    value: OptimizationValue = variable_value

                if value.type == OptimizationValueFunction.__name__:
                    variable_value = value.data.model_dump(mode="json")
                else:
                    variable_value = value.data

                transformed_x: Dict[str, Any] = {
                    "variable_id": variables[variable_id].id,
                    "variable_type": variable.type,
                    "variable_value": variable_value,
                    "variable_value_type": value.type,
                    "client_name": client.name,
                }
                list_transformed_x.append(transformed_x)

            list_dict_x.append(list_transformed_x)

            dict_f: Dict[str, Any] = {}
            dict_f["decision"] = index == decision_index
            for index, f_value in enumerate(f):
                dict_f[f"f{index + 1}"] = f_value

            list_dict_f.append(dict_f)

        f_df: pd.DataFrame = pd.DataFrame(list_dict_f)

        with plot_placeholder.container():
            st.subheader("Objective Space")
            st.pyplot(plots["scatter"].fig)
            st.pyplot(plots["pcp"].fig)
            st.subheader("Solution Space")
            st.dataframe(f_df, height=500)
            st.json(list_dict_x, expanded=False)

        st.session_state["old_result_caches"].add(cache)

    session.close()
    time.sleep(0.01)
