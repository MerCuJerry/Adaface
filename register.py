from pathlib import Path
from face_hnfnu.Config import server_config as config
from httpx import AsyncClient
import asyncio

async def register():
    url=f"http://{config.WEB_SERVER_HOST}:{config.WEB_SERVER_PORT}"
    DIR_PATH = Path.cwd() / "test_reg"
    for img_path in DIR_PATH.iterdir():
        for img in img_path.glob("*.jpg"):
            async with AsyncClient() as client:
                await client.post(f"{url}/add_face", files={"file": (img.as_posix(), img.read_bytes(), "image/jpeg")})

if __name__ == "__main__":
    asyncio.run(register())