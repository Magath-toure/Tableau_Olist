from fastapi import FastAPI
# creation d'une instance fastapi
app = FastAPI()

# définition d'une route  avec une methode
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# definition d'une route avec une variable
@app.get("/items/{item_id}")
def read_item(item_id: int, query_param: str = None):
 
    return {"item_id": item_id, "query_param": query_param}
@app.get("/data/")
def read_data():
    return ()
data = {'categorie'}
