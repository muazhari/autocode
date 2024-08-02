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
	productController := &OneController{
		Router:       router.PathPrefix("/products").Subrouter(),
		OneDatastore: oneDatastore,
	}
	productController.Router.HandleFunc("/searches", productController.SearchMany).Methods(http.MethodGet)
	return productController
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
