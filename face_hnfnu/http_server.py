from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    UploadFile,
    File,
    WebSocket,
    WebSocketDisconnect,
)
import concurrent.futures as futures
from asyncio import get_event_loop
from fastapi.security import HTTPAuthorizationCredentials, HTTPBasic
from fastapi.responses import HTMLResponse
from face_hnfnu.__init__ import adaface, procpool
from face_hnfnu.log import logger
from face_hnfnu.Config import server_config as config
from PIL import Image
import io

app = FastAPI(
    title="AdaFace API",
    description="API for AdaFace",
    version="0.0.1",
)  # create a FastAPI app


@app.get("/live")  # define a route for the live probe
async def _live():
    """服务存活探针接口"""
    return {"result": "OK"}


@app.websocket("/ws/{client_id}")  # define a websocket route for the face recognition
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    logger.info(f"websocket connected with client_id: {client_id}")
    try:
        while True:
            try:
                data = await websocket.receive_bytes()
                image = Image.open(io.BytesIO(data))
                with futures.ThreadPoolExecutor() as pool:
                    thisresult = await get_event_loop().run_in_executor(pool, adaface.verify_face, image, config.SIMILARITY_THRESHOLD)
                if thisresult is None:
                    await websocket.send_json(
                        {"result": "False", "error": "No similar face found"}
                    )
                elif thisresult is not None and thisresult[0] is not None:
                    await websocket.send_json(
                        {
                            "result": "True",
                            "most_similar_face": thisresult[0],
                            "distance": thisresult[1],
                        }
                    )
                else:
                    raise thisresult[1]
            except WebSocketDisconnect as err:
                raise WebSocketDisconnect(reason=str(err))
            except ValueError as err:
                await websocket.send_json(
                    {"result": "False", "error": f"{str(err)}"}
                )
            except Exception as err:
                logger.error(f"verify face failed with error: {str(err)}")
                await websocket.send_json(
                    {"result": "False", "error": f"{str(err)}"}
                )
    except WebSocketDisconnect:
        logger.info("websocket disconnected")

@app.post("/verify")  # verify a face image
async def _verify(file: UploadFile = File()):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        with futures.ThreadPoolExecutor() as pool:
            thisresult = await get_event_loop().run_in_executor(pool, adaface.verify_face, image, config.SIMILARITY_THRESHOLD)
        if thisresult is None:
            return {"result": "False", "error": "No similar face found"}
        elif thisresult is not None and thisresult[0] is not None:
            return {
                "result": "True",
                "most_similar_face": thisresult[0],
                "distance": thisresult[1],
            }
        else:
            raise thisresult[1]
    except Exception as err:
        logger.error(f"verify face failed with error: {str(err)}")
        return {"result": "False", "error": str(err)}


@app.post("/add_face")  # add a face image to the database
async def _add_face(file: UploadFile = File()):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        vector = adaface.ada_face_feature.byte_get_represent(image)
        adaface.face_database.addFace(file.filename, vector.detach().numpy())
        logger.info("add face success")
        return {"result": "True"}
    except Exception as err:
        logger.error(f"add face failed with error: {str(err)}")
        return {"result": "False", "error": str(err)}


@app.post("/remove_face")  # remove a face image from the database
async def _remove_face(face_id: str):
    try:
        adaface.face_database.removeFaceById(face_id)
        logger.info("remove face success")
        return {"result": "True"}
    except Exception as err:
        logger.error(f"remove face failed with error: {str(err)}")
        return {"result": "False", "error": str(err)}
