from pathlib import Path
from pydantic import BaseModel, Field

# Paths
FAISS_DATABASE_PATH = Path.cwd() / "data" / "face_db.index"
INDEX_DATABASE_PATH = Path.cwd() / "data" / "face_db.sqlite"
SERVER_CONFIG_PATH = Path.cwd() / "config" / "config.json"


class ConfigModel(BaseModel):
    WEB_SERVER_PORT: int = Field(default=8000, alias="server_port")
    WEB_SERVER_HOST: str = Field(default="127.0.0.1", alias="server_host")
    ADAFACE_MODEL: str = Field(default="ir_18", alias="adaface_model")
    ADAFACE_MODEL_FILE: str = Field(
        default="models/adaface_ir18_webface4m.ckpt", alias="adaface_model_file"
    )
    FAISS_DATABASE_PATH: str = Field(
        default=FAISS_DATABASE_PATH.as_posix(), alias="faiss_database_path"
    )
    INDEX_DATABASE_PATH: str = Field(
        default=INDEX_DATABASE_PATH.as_posix(), alias="index_database_path"
    )
    SIMILARITY_THRESHOLD: float = Field(default=-1000, alias="threshold")
    THREAD_COUNT: int = Field(default=4, alias="thread_count")


server_config = ConfigModel().model_validate_json(SERVER_CONFIG_PATH.read_text())
