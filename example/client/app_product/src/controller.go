package src

import (
	"encoding/json"
	"github.com/gorilla/mux"
	"net/http"
	"strconv"
	"strings"
)

type OneController struct {
	Router       *mux.Router
	OneDatastore *OneDatastore
}

func NewOneController(router *mux.Router, oneDatastore *OneDatastore) *OneController {
	oneController := &OneController{
		Router:       router.PathPrefix("/products").Subrouter(),
		OneDatastore: oneDatastore,
	}
	oneController.Router.HandleFunc("/searches", oneController.SearchMany).Methods(http.MethodGet)
	router.HandleFunc("/health", oneController.HealthCheck).Methods(http.MethodGet)
	return oneController
}

func (self *OneController) HealthCheck(writer http.ResponseWriter, request *http.Request) {
	writer.WriteHeader(http.StatusOK)
}

func (self *OneController) SearchMany(writer http.ResponseWriter, request *http.Request) {
	keyword := request.URL.Query().Get("keyword")
	topK := request.URL.Query().Get("topK")
	topKInt, _ := strconv.Atoi(topK)
	retrievedProducts := make([]*Product, 0)
	for _, product := range self.OneDatastore.Products {
		if len(retrievedProducts) == topKInt {
			break
		}
		if strings.Contains(keyword, product.Id) || strings.Contains(keyword, product.Name) {
			retrievedProducts = append(retrievedProducts, product)
		}
	}
	writer.Header().Set("Content-Type", "application/json")
	responseBody := &Response[[]*Product]{Data: retrievedProducts}
	_ = json.NewEncoder(writer).Encode(responseBody)
}
