package src

type OneDatastore struct {
	Products []*Product
}

func NewOneDatastore() *OneDatastore {
	return &OneDatastore{
		Products: make([]*Product, 0),
	}
}
