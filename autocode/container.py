from dependency_injector import providers
from dependency_injector.containers import DeclarativeContainer

from autocode.controller import OptimizationController, HealthController
from autocode.datastore import OneDatastore
from autocode.gateway import EvaluationGateway
from autocode.router import ApiRouter
from autocode.setting import ApplicationSetting
from autocode.use_case import OptimizationUseCase, LlmUseCase


class SettingContainer(DeclarativeContainer):
    application = providers.Singleton(
        ApplicationSetting
    )


class GatewayContainer(DeclarativeContainer):
    evaluation = providers.Singleton(
        EvaluationGateway,
    )


class DatastoreContainer(DeclarativeContainer):
    settings = providers.DependenciesContainer()

    one = providers.Singleton(
        OneDatastore,
        application_setting=settings.application
    )


class UseCaseContainer(DeclarativeContainer):
    settings = providers.DependenciesContainer()
    gateways = providers.DependenciesContainer()
    datastores = providers.DependenciesContainer()

    llm = providers.Singleton(
        LlmUseCase,
        application_setting=settings.application
    )

    optimization = providers.Singleton(
        OptimizationUseCase,
        llm_use_case=llm,
        evaluation_gateway=gateways.evaluation,
        one_datastore=datastores.one,
        application_setting=settings.application
    )


class ControllerContainer(DeclarativeContainer):
    use_cases = providers.DependenciesContainer()

    optimization = providers.Singleton(
        OptimizationController,
        optimization_use_case=use_cases.optimization
    )

    health = providers.Singleton(
        HealthController,
    )


class RouterContainer(DeclarativeContainer):
    controllers = providers.DependenciesContainer()

    api = providers.Singleton(
        ApiRouter,
        optimization_controller=controllers.optimization,
        health_controller=controllers.health
    )


class ApplicationContainer(DeclarativeContainer):
    settings = providers.Container(
        SettingContainer
    )
    gateways = providers.Container(
        GatewayContainer,
    )
    datastores = providers.Container(
        DatastoreContainer,
        settings=settings
    )
    use_cases = providers.Container(
        UseCaseContainer,
        settings=settings,
        datastores=datastores,
        gateways=gateways,
    )
    controllers = providers.Container(
        ControllerContainer,
        use_cases=use_cases
    )
    routers = providers.Container(
        RouterContainer,
        controllers=controllers
    )
