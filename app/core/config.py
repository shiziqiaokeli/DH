from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field

class Settings(BaseSettings):
    # 1. 项目基础
    PROJECT_NAME: str = "DH"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    # 2. LLM 配置
    LLM_API_KEY: str
    LLM_BASE_URL: str

    # 3. 数据库配置
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: int = 3306
    DB_NAME: str

    # 4. Redis配置
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str
    REDIS_DB: int

    # 5. 向量数据库配置
    VECTOR_DB_PATH: str 

    # 6. GPT-SOVITS配置
    GPT_SOVITS_URL: str

    # 使用 Pydantic 的 SettingsConfigDict 自动读取 .env
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8')

    # 动态合成数据库 URL (异步驱动使用 aiomysql)
    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        return f"mysql+aiomysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    # 动态合成 Redis URL
    @computed_field
    @property
    def REDIS_URL(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

# 实例化全局配置对象
settings = Settings()