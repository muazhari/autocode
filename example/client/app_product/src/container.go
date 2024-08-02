package src

import "github.com/gorilla/mux"

type DatastoreContainer struct {
	One *OneDatastore
}

func NewDatastoreContainer() *DatastoreContainer {
	return &DatastoreContainer{
		One: NewOneDatastore(),
	}
}

type ControllerContainer struct {
	MainRouter *mux.Router
	One        *OneController
}

func NewControllerContainer(datastoreContainer *DatastoreContainer) *ControllerContainer {
	mainRouter := mux.NewRouter()
	return &ControllerContainer{
		MainRouter: mainRouter,
		One:        NewOneController(mainRouter, datastoreContainer.One),
	}
}

type MainContainer struct {
	Datastore  *DatastoreContainer
	Controller *ControllerContainer
}

func NewMainContainer() *MainContainer {
	datastoreContainer := NewDatastoreContainer()
	controllerContainer := NewControllerContainer(datastoreContainer)
	return &MainContainer{
		Datastore:  datastoreContainer,
		Controller: controllerContainer,
	}
}
