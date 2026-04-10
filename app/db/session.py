from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.core.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=False)   #创建异步引擎
AsyncSessionLocal = async_sessionmaker(   #创建会话工厂
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
async def get_db() -> AsyncSession:   #依赖注入模式管理会话的生命周期
    async with AsyncSessionLocal() as session:
        yield session