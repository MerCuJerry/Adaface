import uvicorn
import asyncio
import signal
from src.http_server import app
from src.Config import config

async def start_server():
    server_config = uvicorn.Config(app, host="0.0.0.0", port=39040, log_level="info")
    server = uvicorn.Server(server_config)
    await server.serve()
    
async def shutdown_event():
    await config.shutdown_event()
    await asyncio.sleep(1)
    print("Shutdown")

def signal_handler(sig, frame):
    loop = asyncio.get_event_loop()
    loop.create_task(shutdown_event())
    loop.stop()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(config.startup_event())
    loop.run_until_complete(start_server())