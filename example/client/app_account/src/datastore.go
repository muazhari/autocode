package src

type OneDatastore struct {
	Accounts []*Account
}

func NewOneDatastore() *OneDatastore {
	return &OneDatastore{
		Accounts: make([]*Account, 0),
	}
}
