package src

type Response[T any] struct {
	Data T `json:"data"`
}
type Account struct {
	Id       string `json:"id"`
	Email    string `json:"email"`
	Password string `json:"password"`
}
