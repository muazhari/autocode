package src

type Response[T any] struct {
	Data T `json:"data"`
}

type Product struct {
	Id    string  `json:"id"`
	Name  string  `json:"name"`
	Price float64 `json:"price"`
	Stock int     `json:"stock"`
}
