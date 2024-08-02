package test

import (
	"app_account/src"
	"encoding/json"
	"github.com/muazhari/autocode-go"
	"github.com/stretchr/testify/assert"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

type Application struct {
	Test      *testing.T
	Container *src.MainContainer
	Server    *httptest.Server
}

func NewApplication(t *testing.T) *Application {
	return &Application{
		Test:      t,
		Container: nil,
		Server:    nil,
	}
}

func (self *Application) Duplicate(ctx *autocode.Optimization) any {
	container := src.NewMainContainer()
	container.Datastore.One.Accounts = []*src.Account{
		{
			Id:       "id1",
			Email:    "email1",
			Password: "password1",
		},
		{
			Id:       "id2",
			Email:    "email2",
			Password: "password2",
		},
	}
	server := httptest.NewServer(container.Controller.MainRouter)
	return &Application{
		Test:      self.Test,
		Container: container,
		Server:    server,
	}
}

func (self *Application) TestSearchMany(t *testing.T) {
	t.Parallel()

	url := self.Server.URL + "/accounts/searches?keyword=email1&topK=1"
	response, _ := http.Get(url)
	assert.Equal(self.Test, http.StatusOK, response.StatusCode)
	responseBody := &src.Response[[]*src.Account]{}
	_ = json.NewDecoder(response.Body).Decode(responseBody)
	expectedData := []*src.Account{
		self.Container.Datastore.One.Accounts[0],
	}
	assert.Equal(self.Test, expectedData, responseBody.Data)
}

func (self *Application) Evaluate(ctx *autocode.Optimization) *autocode.OptimizationEvaluateRunResponse {
	t0 := time.Now()
	self.Test.Run("TestSearchMany", self.TestSearchMany)
	t1 := time.Now()
	f_sum_latency := float64(0)
	f_sum_latency += float64(t1.Sub(t0).Microseconds())
	f_sum_output := float64(0)
	f_sum_output += float64(ctx.GetValue("a").(int64))
	f_sum_output += float64(ctx.GetValue("b").(int64))
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
			f_sum_latency, f_sum_output,
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
		autocode.NewOptimizationChoice(
			"b",
			[]any{b0, b1, b2},
		),
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
		"localhost",
		11000,
		"app_account",
		[]string{},
	)
	optimization.Prepare()
}
