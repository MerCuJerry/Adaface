from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBasic
from fastapi.responses import HTMLResponse
from .Config import config

app = FastAPI(
    title="AdaFace API",
    description="API for AdaFace",
    version="0.0.1",
) # create a FastAPI app

@app.get("/live") # define a route for the live probe
async def _live():
    """服务存活探针接口
    """
    return {'result': "OK"}

@app.post("/verify") # verify a face image
async def _verify(): 
    query_vector = config.ada_face_feature.byte_get_represent("img")
    thisresult = config.pool.apply_async(config.face_database.searchSimilarFaces, args=(query_vector.detach().numpy()[0], 0.75)).get()
    return {"result":"True", "most_similar_face": thisresult[0]}

@app.post("/add_face") # add a face image to the database
async def _add_face():
    pass

@app.post("/remove_face") # remove a face image from the database
async def _remove_face():
    pass
