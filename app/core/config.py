from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field

class Settings(BaseSettings):
    #1.LLM
    LLM_API_KEY: str
    LLM_BASE_URL: str
    #2.MySQL
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: int = 3306
    DB_NAME: str
    #3.Redis
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str
    REDIS_DB: int
    #4.ChromaDB
    VECTOR_DB_PATH: str 
    #5.GPT-SOVITS
    TTS_URL: str
    TRAIN_URL: str
 
    model_config = SettingsConfigDict(   #使用Pydantic的SettingsConfigDict自动读取.env
        env_file=".env", 
        env_file_encoding='utf-8',
        )
    @computed_field   #动态合成MySQL URL(异步驱动使用aiomysql)
    @property
    def DATABASE_URL(self) -> str:
        return f"mysql+aiomysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    @computed_field   #动态合成Redis URL
    @property
    def REDIS_URL(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
settings = Settings()   #实例化全局配置对象