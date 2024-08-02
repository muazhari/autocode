package src

type Response[T any] struct {
	Data T `json:"data"`
}
type Account struct {
	Id       string
	Email    string
	Password string
}
