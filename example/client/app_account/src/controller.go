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
	accountController := &OneController{
		Router:       router.PathPrefix("/accounts").Subrouter(),
		OneDatastore: oneDatastore,
	}
	accountController.Router.HandleFunc("/searches", accountController.SearchMany).Methods(http.MethodGet)
	return accountController
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
