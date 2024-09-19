import uvicorn
import asyncio
from face_hnfnu.http_server import app
from face_hnfnu.log import logger
from face_hnfnu.__init__ import adaface, procpool
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


if __name__ == "__main__":
    try:
        adaface.startup_event()
        #procpool.startup_event()
        asyncio.run(start_server())
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Server Shutdown")
        #procpool.shutdown_event()
        adaface.shutdown_event()
