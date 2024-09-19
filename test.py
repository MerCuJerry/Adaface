from pathlib import Path
from PIL import Image
from face_hnfnu.Config import server_config as config
from face_hnfnu.log import logger
from httpx import AsyncClient
import websockets
import io
import asyncio
import time

async def test():
    url=f"http://{config.WEB_SERVER_HOST}:{config.WEB_SERVER_PORT}"
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
        async with AsyncClient() as client:
            for i in range(100):
                response = await client.post(f"{url}/verify", files={"file": (img_path.as_posix(), img_path.read_bytes(), "image/jpeg")})
                logger.warning(f"The most similar face of {img_path.as_posix()} is {response.json()['most_similar_face']} with distance {response.json()['distance']}")
    end_time = time.time()
    logger.warning(f"Finished stress test in {end_time - start_time} seconds")

async def test_stress_only():
    url=f"http://{config.WEB_SERVER_HOST}:{config.WEB_SERVER_PORT}"

    #压力测试
    img_path = Path.cwd() / "test" / "test.jpg"
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
        async with AsyncClient() as client:
            for i in range(100):
                response = await client.post(f"{url}/verify", files={"file": (img_path.as_posix(), img_path.read_bytes(), "image/jpeg")})
                logger.warning(f"The most similar face of {img_path.as_posix()} is {response.json()['most_similar_face']} with distance {response.json()['distance']}")
    '''
        task = [verify(url,img_path) for i in range(100)]
    for task in asyncio.as_completed(task):
        result = await task
        logger.info(f"The most similar face of {img_path.as_posix()} is {result.json()['most_similar_face']} with distance {result.json()['distance']}")
    '''

    end_time = time.time()
    logger.warning(f"Finished stress test in {end_time - start_time} seconds")

async def verify(url, img_path):
    async with asyncio.Semaphore(10):
        async with AsyncClient() as client:
            response = await client.post(f"{url}/verify", files={"file": (img_path.as_posix(), img_path.read_bytes(), "image/jpeg")})
    return response

if __name__ == '__main__':
    #signal.signal(signal.SIGINT, signal_handler)
    #启动服务并测试
    loop = asyncio.new_event_loop()
    loop.run_until_complete(test_stress_only())
    loop.stop()