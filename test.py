from imutils import paths
from pathlib import Path
from face_hnfnu.__main__  import start_server, shutdown_event
from face_hnfnu.__init__  import adaface
from httpx import AsyncClient
import asyncio

async def test():

    await asyncio.sleep(2)
    DIR_PATH = Path.cwd() / "test_reg"
    for img_path in paths.list_images(DIR_PATH):
        print(img_path)
        with open(img_path, "rb") as f:
            async with AsyncClient() as client:
                response = await client.post("http://127.0.0.1:30035/add_face", files={"file": (img_path, f, "image/jpeg")})
                await asyncio.sleep(1)
                print(response.json())
    
    #图片检测测试
    TEST_PATH = Path.cwd() / "test"
    threshold = -1000
    for img_path in paths.list_images(TEST_PATH):
        with open(img_path, "rb") as f:
            async with AsyncClient() as client:
                response = await client.post("http://127.0.0.1:30035/verify", files={"file": (img_path, f, "image/jpeg")})
                await asyncio.sleep(1)
        print(f"The most similar face of {img_path} is {response.json()}")

if __name__ == '__main__':
    DB = Path.cwd() / "data" / "face_db.sqlite"
    DB.rename(DB.with_name("bak"))
    loop = asyncio.new_event_loop()
    loop.run_until_complete(adaface.startup_event())
    loop.create_task(start_server())
    loop.run_until_complete(test())
    #处理过的人脸照片，以做面部对齐处理,大小： 112*112
    #图片注册
    loop.create_task(shutdown_event())
    loop.stop()