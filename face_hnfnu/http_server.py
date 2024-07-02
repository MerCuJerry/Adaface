from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.security import HTTPAuthorizationCredentials, HTTPBasic
from fastapi.responses import HTMLResponse
from face_hnfnu.__init__ import adaface
from face_hnfnu.log import logger
from PIL import Image
import io

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
async def _verify(threshold: int, file: UploadFile):
    content = await file.read()
    image = Image.open(io.BytesIO(content))
    query_vector = adaface.ada_face_feature.byte_get_represent(image)
    thisresult = adaface.pool.apply_async(adaface.face_database.searchSimilarFaces, args=(query_vector.detach().numpy()[0], threshold)).get()
    if thisresult is not None:
        return {"result":"True", "most_similar_face": thisresult[0]}
    else:
        return {"result":"False"}

@app.post("/add_face") # add a face image to the database
async def _add_face(file: UploadFile):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        vector = adaface.ada_face_feature.byte_get_represent(image)
        adaface.face_database.addFace(file.filename, vector.detach().numpy()[0])
        logger.info("add face success")
        return {"result":"True"}
    except Exception as e:
        return {"result":"False", "error": str(e)}
    

@app.post("/remove_face") # remove a face image from the database
async def _remove_face():
    pass
