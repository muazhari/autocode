package test

import (
	"app_product/src"
	"encoding/json"
	"fmt"
	"github.com/muazhari/autocode-go"
	"github.com/stretchr/testify/assert"
	"io"
	"net/http"
	"os"
	"testing"
	"time"
)

func waitServer(method string, url string, body io.Reader, timeout time.Duration) error {
	ch := make(chan bool)
	go func() {
		for {
			client := http.Client{}
			request, _ := http.NewRequest(method, url, body)
			_, responseErr := client.Do(request)
			if responseErr == nil {
				ch <- true
			}
			time.Sleep(10 * time.Millisecond)
		}
	}()

	select {
	case <-ch:
		return nil
	case <-time.After(timeout):
		return fmt.Errorf("server did not reply after %v", timeout)
	}
}

type Application struct {
	Test      *testing.T
	Container *src.MainContainer
}

func NewApplication(t *testing.T) *Application {
	container := src.NewMainContainer()
	go func() {
		address := fmt.Sprintf("0.0.0.0:%s", os.Getenv("GATEWAY_PORT"))
		err := http.ListenAndServe(address, container.Controller.MainRouter)
		if err != nil {
			panic(err)
		}
	}()
	waitServerErr := waitServer(
		http.MethodGet,
		fmt.Sprintf("http://localhost:%s/health", os.Getenv("GATEWAY_PORT")),
		nil,
		3*time.Second,
	)
	if waitServerErr != nil {
		panic(waitServerErr)
	}
	return &Application{
		Test:      t,
		Container: container,
	}
}

func (self *Application) AccountTestSearchMany(t *testing.T) {
	t.Parallel()

	url := fmt.Sprintf(
		"http://%s:%s/accounts/searches?keyword=email1&topK=1",
		os.Getenv("ACCOUNT_HOST"),
		os.Getenv("ACCOUNT_PORT"),
	)
	response, _ := http.Get(url)
	assert.Equal(self.Test, http.StatusOK, response.StatusCode)
	responseBody := &src.Response[[]*src.Account]{}
	_ = json.NewDecoder(response.Body).Decode(responseBody)
	expectedData := []*src.Account{
		{
			Id:       "id1",
			Email:    "email1",
			Password: "password1",
		},
	}
	assert.Equal(self.Test, expectedData, responseBody.Data)
}

func (self *Application) ProductTestSearchMany(t *testing.T) {
	t.Parallel()

	url := fmt.Sprintf(
		"http://%s:%s/products/searches?keyword=name1&topK=1",
		os.Getenv("PRODUCT_HOST"),
		os.Getenv("PRODUCT_PORT"),
	)
	response, _ := http.Get(url)
	assert.Equal(self.Test, http.StatusOK, response.StatusCode)
	responseBody := &src.Response[[]*src.Product]{}
	_ = json.NewDecoder(response.Body).Decode(responseBody)
	expectedData := []*src.Product{
		{
			Id:    "id1",
			Name:  "name1",
			Price: 1.0,
		},
	}
	assert.Equal(self.Test, expectedData, responseBody.Data)
}

func (self *Application) Evaluate(ctx *autocode.Optimization) *autocode.OptimizationEvaluateRunResponse {
	t0 := time.Now()
	self.Test.Run("AccountTestSearchMany", self.AccountTestSearchMany)
	self.Test.Run("ProductTestSearchMany", self.ProductTestSearchMany)
	t1 := time.Now()
	f_avg_latency := float64(0)
	f_avg_latency += float64(t1.Sub(t0).Microseconds()) / 2

	return &autocode.OptimizationEvaluateRunResponse{
		Objectives: []float64{
			f_avg_latency,
		},
		InequalityConstraints: []float64{},
		EqualityConstraints:   []float64{},
	}
}

func Test(t *testing.T) {
	application := NewApplication(t)

	variables := []any{}
	optimization := autocode.NewOptimization(
		variables,
		application,
		"host.docker.internal",
		10000,
		11000,
	)
	optimization.Prepare()
}
