import os

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {
        "worker_id": os.environ.get("WORKER_ID"),
        "app_name": os.environ.get("APP_NAME"),
    }
