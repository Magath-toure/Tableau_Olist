from fastapi import FastAPI
import numpy as np
from train_model import make_model_save
from make_pred import make_prediction
import pandas as pd

app = FastAPI()

@app.get("/infos")
def read_root():
    return {"message": "Hello, welcome on my dashboard!"}
