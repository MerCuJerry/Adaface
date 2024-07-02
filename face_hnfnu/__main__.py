import uvicorn
import asyncio
import signal
from face_hnfnu.http_server import app
from face_hnfnu.log import logger
from face_hnfnu.__init__ import adaface
from face_hnfnu.Config import server_config as config


async def start_server():
    logger.info("Server Startup")
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "handlers": {
            "default": {
                "class": "face_hnfnu.log.LoguruHandler",
            },
        },
        "loggers": {
            "uvicorn.error": {"handlers": ["default"], "level": "INFO"},
            "uvicorn.access": {
                "handlers": ["default"],
                "level": "INFO",
            },
        },
    }
    server_config = uvicorn.Config(
        app,
        host=config.WEB_SERVER_HOST,
        port=config.WEB_SERVER_PORT,
        log_config=LOGGING_CONFIG,
        log_level="info",
    )
    server = uvicorn.Server(server_config)
    await server.serve()


async def shutdown_event():
    await adaface.shutdown_event()
    logger.info("Server Shutdown")


def signal_handler(sig, frame):
    loop = asyncio.get_event_loop()
    loop.create_task(shutdown_event())
    loop.stop()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(adaface.startup_event())
    loop.run_until_complete(start_server())
