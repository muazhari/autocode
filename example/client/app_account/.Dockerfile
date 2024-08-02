FROM golang:latest

WORKDIR /workdir

COPY . .

RUN go mod tidy