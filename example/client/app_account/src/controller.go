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
		Router:       router.PathPrefix("/accounts").Subrouter(),
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
	retrievedAccounts := make([]*Account, 0)
	for _, account := range self.OneDatastore.Accounts {
		if len(retrievedAccounts) == topKInt {
			break
		}
		if strings.Contains(keyword, account.Id) || strings.Contains(keyword, account.Email) || strings.Contains(keyword, account.Password) {
			retrievedAccounts = append(retrievedAccounts, account)
		}
	}
	writer.Header().Set("Content-Type", "application/json")
	responseBody := &Response[[]*Account]{Data: retrievedAccounts}
	_ = json.NewEncoder(writer).Encode(responseBody)
}
