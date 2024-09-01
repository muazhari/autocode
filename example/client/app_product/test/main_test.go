package test

import (
	"app_product/src"
	"encoding/json"
	"fmt"
	"github.com/muazhari/autocode-go"
	"github.com/stretchr/testify/assert"
	"io"
	"math"
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
	container.Datastore.One.Products = []*src.Product{
		{
			Id:    "id1",
			Name:  "name1",
			Price: 1.0,
		},
		{
			Id:    "id2",
			Name:  "name2",
			Price: 2.0,
		},
	}
	go func() {
		address := fmt.Sprintf("0.0.0.0:%s", os.Getenv("PRODUCT_PORT"))
		err := http.ListenAndServe(address, container.Controller.MainRouter)
		if err != nil {
			panic(err)
		}
	}()
	waitServerErr := waitServer(
		http.MethodGet,
		fmt.Sprintf("http://localhost:%s/health", os.Getenv("PRODUCT_PORT")),
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

func (self *Application) TestSearchMany(t *testing.T) {
	t.Parallel()

	url := fmt.Sprintf(
		"http://localhost:%s/products/searches?keyword=name1&topK=1",
		os.Getenv("PRODUCT_PORT"),
	)
	response, _ := http.Get(url)
	assert.Equal(self.Test, http.StatusOK, response.StatusCode)
	responseBody := &src.Response[[]*src.Product]{}
	_ = json.NewDecoder(response.Body).Decode(responseBody)
	expectedData := []*src.Product{
		self.Container.Datastore.One.Products[0],
	}
	assert.Equal(self.Test, expectedData, responseBody.Data)
}

func (self *Application) Evaluate(ctx *autocode.Optimization) *autocode.OptimizationEvaluateRunResponse {
	f_sum_output := float64(0)
	f_sum_output += float64(ctx.GetValue("a").(int64))
	//f_sum_output += float64(ctx.GetValue("b").(int64))
	//f_sum_output += float64(ctx.GetValue("c").(int64))
	//f_sum_output += float64(ctx.GetValue("d").(int64))
	//f_sum_output += ctx.GetValue("e").(float64)
	//if ctx.GetValue("f").(bool) {
	//	f_sum_output += 1.0
	//}
	f_sum_understandability := float64(0)
	f_sum_complexity := float64(0)
	f_sum_readability := float64(0)
	f_sum_error_potentiality := float64(0)
	f_sum_overall_maintaianability := float64(0)
	variable_function_count := float64(0)

	for variableId, variableValue := range ctx.VariableValues {
		if variableValue.Type == autocode.VALUE_FUNCTION {
			variable := ctx.Variables[variableId]
			choice := variable.(*autocode.OptimizationChoice)
			option := choice.Options[variableValue.Id]
			function := option.Data.(*autocode.OptimizationFunctionValue)
			f_sum_understandability += function.Understandability
			f_sum_complexity += function.Complexity
			f_sum_readability += function.Readability
			f_sum_error_potentiality += function.ErrorPotentiality
			f_sum_overall_maintaianability += function.OverallMaintainability
			variable_function_count += 1
		}
	}

	return &autocode.OptimizationEvaluateRunResponse{
		Objectives: []float64{
			f_sum_output,
			f_sum_understandability / variable_function_count, f_sum_complexity / variable_function_count,
			f_sum_readability / variable_function_count, f_sum_error_potentiality / variable_function_count,
			f_sum_overall_maintaianability / variable_function_count,
		},
		InequalityConstraints: []float64{},
		EqualityConstraints:   []float64{},
	}
}

func a0(ctx *autocode.Optimization, args ...any) (result any) {
	n := int64(1)
	count := int64(0)
	for i := int64(1); i < n; i++ {
		count += 1
	}
	return count
}

func a1(ctx *autocode.Optimization, args ...any) (result any) {
	n := int64(1)
	count := int64(0)
	for i := int64(1); i < n; i++ {
		for j := int64(1); j < n; j++ {
			count += 1
		}
	}
	return count
}

func b0(ctx *autocode.Optimization, args ...any) (result any) {
	n := ctx.GetValue("a").(int64)
	count := int64(0)
	for i := int64(0); i < n/2; i++ {
		count += 1
	}
	return count
}

func b1(ctx *autocode.Optimization, args ...any) (result any) {
	n := ctx.GetValue("a").(int64)
	count := int64(0)
	for i := int64(1); i < n/2; i++ {
		for j := int64(1); j < n/i; j++ {
			count += 1
		}
	}
	return count
}

func b2(ctx *autocode.Optimization, args ...any) (result any) {
	n := int64(1)
	count := int64(0)
	for i := int64(1); i < n; i++ {
		for j := int64(1); j < n; j++ {
			for k := int64(1); k < n; k++ {
				count += 1
			}
		}
	}
	return count
}

func c0(ctx *autocode.Optimization, args ...any) (result any) {
	n := ctx.GetValue("b").(int64)
	count := int64(0)
	for i := int64(1); i < n; i++ {
		for j := int64(1); j < (i * i); j++ {
			count += 1
		}
	}
	return count
}

func c1(ctx *autocode.Optimization, args ...any) (result any) {
	n := int64(1)
	count := int64(0)
	for i := int64(1); i < int64(math.Log2(float64(n))); i++ {
		for j := int64(1); j < n-i; j++ {
			count += 1
		}
	}
	return count
}

func Test(t *testing.T) {
	application := NewApplication(t)

	variables := []any{
		autocode.NewOptimizationChoice(
			"a",
			[]any{a0, a1},
		),
		//autocode.NewOptimizationChoice(
		//	"b",
		//	[]any{b0, b1, b2},
		//),
		//autocode.NewOptimizationChoice(
		//	"c",
		//	[]any{c0, c1},
		//),
		autocode.NewOptimizationInteger(
			"d",
			-10, 10,
		),
		autocode.NewOptimizationReal(
			"e",
			-3.14, 3.14,
		),
		autocode.NewOptimizationBinary(
			"f",
		),
	}
	optimization := autocode.NewOptimization(
		variables,
		application,
		"host.docker.internal",
		10000,
		11000,
	)
	optimization.Prepare()
}
