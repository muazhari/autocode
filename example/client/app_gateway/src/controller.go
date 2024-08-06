package src

import (
	"encoding/json"
	"fmt"
	"github.com/gorilla/mux"
	"net/http"
	"os"
)

type OneController struct {
	Router *mux.Router
}

func NewOneController(router *mux.Router) *OneController {
	oneController := &OneController{
		Router: router.PathPrefix("/gateways").Subrouter(),
	}
	oneController.Router.HandleFunc("/accounts/searches", oneController.AccountSearchMany).Methods(http.MethodGet)
	oneController.Router.HandleFunc("/products/searches", oneController.ProductSearchMany).Methods(http.MethodGet)
	router.HandleFunc("/health", oneController.HealthCheck).Methods(http.MethodGet)
	return oneController
}

func (self *OneController) HealthCheck(writer http.ResponseWriter, request *http.Request) {
	writer.WriteHeader(http.StatusOK)
}

func (self *OneController) AccountSearchMany(writer http.ResponseWriter, request *http.Request) {
	keyword := request.URL.Query().Get("keyword")
	topK := request.URL.Query().Get("topK")
	url := fmt.Sprintf(
		"http://%s:%s/accounts/searches?keyword=%s&topK=%s",
		os.Getenv("ACCOUNT_HOST"),
		os.Getenv("ACCOUNT_PORT"),
		keyword,
		topK,
	)
	response, responseErr := http.Get(url)
	if responseErr != nil {
		writer.WriteHeader(http.StatusInternalServerError)
		return
	}
	writer.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(writer).Encode(response.Body)
}

func (self *OneController) ProductSearchMany(writer http.ResponseWriter, request *http.Request) {
	keyword := request.URL.Query().Get("keyword")
	topK := request.URL.Query().Get("topK")
	url := fmt.Sprintf(
		"http://%s:%s/products/searches?keyword=%s&topK=%s",
		os.Getenv("PRODUCT_HOST"),
		os.Getenv("PRODUCT_PORT"),
		keyword,
		topK,
	)
	response, responseErr := http.Get(url)
	if responseErr != nil {
		writer.WriteHeader(http.StatusInternalServerError)
		return
	}
	writer.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(writer).Encode(response.Body)
}
