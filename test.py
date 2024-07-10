from pathlib import Path
import signal
from PIL import Image
from face_hnfnu.__main__  import start_server, shutdown_event, signal_handler
from face_hnfnu.__init__  import adaface, procpool
from face_hnfnu.Config import server_config as config
from face_hnfnu.log import logger
from httpx import AsyncClient
import websockets
import io
import asyncio
import time

async def test():
    url=f"http://{config.WEB_SERVER_HOST}:{config.WEB_SERVER_PORT}"
    await asyncio.sleep(2)
    #注册测试
    DIR_PATH = Path.cwd() / "test_reg"
    for img_path in DIR_PATH.glob("*.jpg"):
        logger.warning(f"Registering {img_path}")
        async with AsyncClient() as client:
            response = await client.post(f"{url}/add_face", files={"file": (img_path.as_posix(), img_path.read_bytes(), "image/jpeg")})
            await asyncio.sleep(1)

    #图片检测测试
    TEST_PATH = Path.cwd() / "test"
    for img_path in TEST_PATH.glob("*.jpg"):
        async with AsyncClient() as client:
            response = await client.post(f"{url}/verify", files={"file": (img_path.as_posix(), img_path.read_bytes(), "image/jpeg")})
            await asyncio.sleep(1)
        if response.json()['result'] == "True":
            logger.warning(f"The most similar face of {img_path.as_posix()} is {response.json()['most_similar_face']} with distance {response.json()['distance']}")
        else:
            logger.warning(response.json()["error"])
    
    #删除测试
    img_id = Path.cwd() / "test_reg" / "fbeff_109.73.jpg"
    async with AsyncClient() as client:
        response = await client.post(f"{url}/remove_face", params={"face_id": img_id.as_posix()})
        await asyncio.sleep(1)

    async with AsyncClient() as client:
        response = await client.post(f"{url}/verify", files={"file": (img_path.as_posix(), img_path.read_bytes(), "image/jpeg")})
        await asyncio.sleep(1)
    if response.json()['result'] == "True":
        logger.warning(f"The most similar face of {img_path} is {response.json()['most_similar_face']} with distance {response.json()['distance']}")
    else:
        logger.warning(response.json()["error"])
        
    #压力测试
    img_path = Path.cwd() / "test" / "img1.jpg"
    start_time = time.time()
    img = Image.open(io.BytesIO(img_path.read_bytes()))
    w, h = img.size
    if w > 960 or h > 960:
        if w < h:
            aspect_ratio = w / h
            img.thumbnail((960 * aspect_ratio, 960))
        else:
            aspect_ratio = h / w
            img.thumbnail((960, 960 * aspect_ratio))
    with io.BytesIO() as output:
        img.save(output, format="webp")
        async with websockets.connect(f"ws://{config.WEB_SERVER_HOST}:{config.WEB_SERVER_PORT}/ws/test") as client:
            for i in range(100):
                await client.send(output.getvalue())
                response = await client.recv()
                logger.warning(f"Received response: {response}")
    end_time = time.time()
    logger.warning(f"Finished stress test in {end_time - start_time} seconds")

    #压力测试2
    start_time = time.time()
    for i in range(100):
        response =  procpool.pool.apply_async(adaface.verify_face, args=(img,config.SIMILARITY_THRESHOLD)).get()
        logger.warning(f"{response}")
    end_time = time.time()
    logger.warning(f"Finished stress test 2 in {end_time - start_time} seconds")

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    #删除原有数据库
    DB = Path.cwd() / "data" / "face_db.sqlite"
    if DB.exists():
        DB.rename(DB.with_name("bak"))
    #启动服务并测试
    loop = asyncio.new_event_loop()
    loop.run_until_complete(adaface.startup_event())
    loop.run_until_complete(procpool.startup_event())
    loop.create_task(start_server())
    loop.run_until_complete(test())
    loop.create_task(shutdown_event())
    loop.stop()