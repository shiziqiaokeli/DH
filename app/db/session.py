from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.core.config import settings

#创建异步引擎
engine = create_async_engine(settings.DATABASE_URL, echo=False)
#创建会话工厂
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
#依赖注入模式管理会话的生命周期
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session