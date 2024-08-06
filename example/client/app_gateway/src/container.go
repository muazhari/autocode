package src

import "github.com/gorilla/mux"

type ControllerContainer struct {
	MainRouter *mux.Router
	One        *OneController
}

func NewControllerContainer() *ControllerContainer {
	mainRouter := mux.NewRouter()
	return &ControllerContainer{
		MainRouter: mainRouter,
		One:        NewOneController(mainRouter),
	}
}

type MainContainer struct {
	Controller *ControllerContainer
}

func NewMainContainer() *MainContainer {
	controllerContainer := NewControllerContainer()
	return &MainContainer{
		Controller: controllerContainer,
	}
}
