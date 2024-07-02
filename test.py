from imutils import paths
from pathlib import Path
from face_hnfnu.__main__  import start_server, shutdown_event
from face_hnfnu.__init__  import adaface
from httpx import AsyncClient
import asyncio

async def test():

    await asyncio.sleep(2)
    #注册测试
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
    for img_path in paths.list_images(TEST_PATH):
        with open(img_path, "rb") as f:
            async with AsyncClient() as client:
                response = await client.post("http://127.0.0.1:30035/verify", files={"file": (img_path, f, "image/jpeg")})
                await asyncio.sleep(1)
        print(f"The most similar face of {img_path} is {response.json()}")

if __name__ == '__main__':
    #删除原有数据库
    DB = Path.cwd() / "data" / "face_db.sqlite"
    if DB.exists():
        DB.rename(DB.with_name("bak"))
    #启动服务并测试
    loop = asyncio.new_event_loop()
    loop.run_until_complete(adaface.startup_event())
    loop.create_task(start_server())
    loop.run_until_complete(test())
    loop.create_task(shutdown_event())
    loop.stop()