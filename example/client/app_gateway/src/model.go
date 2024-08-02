package src

type Response[T any] struct {
	Data T `json:"data"`
}
type Product struct {
	Id    string
	Name  string
	Price float64
	Stock int
}
