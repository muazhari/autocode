import hashlib
import uuid
from typing import Optional, Any, List, Dict, Tuple, TypedDict

import dill
import numpy as np
from pydantic import BaseModel as PydanticBaseModelV2, ConfigDict, Field, PrivateAttr
from pydantic.v1 import BaseModel as BaseModelV1, Field as FieldV1
from pymoo.core.plot import Plot
from sqlmodel import SQLModel, Field as SQLField


class BaseModel(PydanticBaseModelV2):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        revalidate_instances="always",
        validate_default=True,
        validate_return=True,
        validate_assignment=True,
    )


class CodeScoring(BaseModelV1):
    """
    Score code in based on the following statements:
    Error Potentiality - this code is potentially error-prone;
    Readability - this code is easy to read;
    Understandability - the semantic meaning of this code is clear;
    Complexity - this code is complex;
    Modularity  - this code should be broken into smaller pieces;
    Overall maintainability - overall, this code is maintainable.
    The score scale from 1 (strongly agree) to 100 (strongly disagree).
    You must score in precision, i.e. 14.3, 47.456, 75.45, 58.58495, 3.141598, etc.
    """
    error_potentiality: float = FieldV1(description="Error potential score.")
    readability: float = FieldV1(description="Readability score.")
    understandability: float = FieldV1(description="Understandability score.")
    complexity: float = FieldV1(description="Complexity score.")
    modularity: float = FieldV1(description="Modularity score.")
    overall_maintainability: float = FieldV1(description="Overall maintainability score.")


class CodeVariation(BaseModelV1):
    """
    Code variation is a code snippet that is a variation of the original code.
    """
    variation: Optional[str] = FieldV1(description="Code variation.", default=None)


class ScoringState(TypedDict):
    code: str
    analysis: str
    score: List[CodeScoring]


class VariationState(TypedDict):
    code: str
    analysis: str
    variation: List[CodeVariation]
    new_function_name: str


class OptimizationVariable(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    name: str
    _client_id: Optional[str] = PrivateAttr(default=None)

    def get_client_id(self):
        return self._client_id

    def set_client_id(self, client_id: str):
        self._client_id = client_id


class OptimizationValueFunction(BaseModel):
    name: str
    string: str
    error_potentiality: Optional[float] = Field(default=None)
    understandability: Optional[float] = Field(default=None)
    complexity: Optional[float] = Field(default=None)
    readability: Optional[float] = Field(default=None)
    modularity: Optional[float] = Field(default=None)
    overall_maintainability: Optional[float] = Field(default=None)


class OptimizationValue(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: Optional[str]
    data: Any | OptimizationValueFunction

    @staticmethod
    def convert_type(data: Any):
        data_type = type(data)
        if data_type == np.float64:
            return float(data)
        elif data_type == np.int64:
            return int(data)
        elif data_type == np.bool_:
            return bool(data)
        else:
            return data

    def __init__(self, **data):
        data["data"] = self.convert_type(data["data"])
        if data.get("type", None) == OptimizationValueFunction.__name__:
            if type(data["data"]) != OptimizationValueFunction:
                data["data"] = OptimizationValueFunction(**data["data"])
        data["type"] = type(data["data"]).__name__
        super().__init__(**data)


class OptimizationBinary(OptimizationVariable):
    pass


class OptimizationChoice(OptimizationVariable):
    options: Dict[str, OptimizationValue]


class OptimizationReal(OptimizationVariable):
    bounds: Tuple[float, float]


class OptimizationInteger(OptimizationVariable):
    bounds: Tuple[int, int]


class OptimizationObjective(BaseModel):
    type: str


class OptimizationClient(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    variables: Dict[str, OptimizationBinary | OptimizationChoice | OptimizationInteger | OptimizationReal]
    name: str
    host: str
    port: int


class OptimizationPrepareRequest(BaseModel):
    variables: Dict[str, OptimizationBinary | OptimizationChoice | OptimizationInteger | OptimizationReal]
    host: str
    port: int
    name: str

    def __init__(self, **data):
        transformed_variables: Dict[str, OptimizationVariable] = {}
        for variable_id, variable in data["variables"].items():
            if variable["type"] == OptimizationBinary.__name__:
                transformed_variables[variable_id] = OptimizationBinary(**variable)
            elif variable["type"] == OptimizationChoice.__name__:
                transformed_variables[variable_id] = OptimizationChoice(**variable)
            elif variable["type"] == OptimizationInteger.__name__:
                transformed_variables[variable_id] = OptimizationInteger(**variable)
            elif variable["type"] == OptimizationReal.__name__:
                transformed_variables[variable_id] = OptimizationReal(**variable)
            else:
                raise ValueError(f"Variable type {variable['type']} is not supported.")
        data["variables"] = transformed_variables
        super().__init__(**data)


class OptimizationPrepareResponse(BaseModel):
    variables: Dict[str, OptimizationBinary | OptimizationChoice | OptimizationInteger | OptimizationReal]
    num_workers: int


class OptimizationEvaluatePrepareRequest(BaseModel):
    worker_id: str
    variable_values: Dict[str, OptimizationValue]


class OptimizationEvaluateRunRequest(BaseModel):
    worker_id: str


class OptimizationEvaluateRunResponse(BaseModel):
    objectives: List[float]
    inequality_constraints: List[float]
    equality_constraints: List[float]


class OptimizationInterpretation(BaseModel):
    objectives: List[List[float]]
    solutions: List[Dict[str, OptimizationValue]]
    decision_index: int
    plots: List[Plot]


class Cache(SQLModel, table=True):
    key: str = SQLField(primary_key=True)
    value: bytes

    def __hash__(self):
        return int.from_bytes(
            bytes=hashlib.sha256(dill.dumps(self)).digest(),
            byteorder="big"
        )
