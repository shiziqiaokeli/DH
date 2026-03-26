import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy import String,Float
from sqlalchemy import func
from datetime import datetime
from sqlalchemy import DateTime
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import DeclarativeBase,Mapped,mapped_column
from sqlalchemy.ext.asyncio import async_sessionmaker,AsyncSession
from sqlalchemy import select
from fastapi import Depends
from pydantic import BaseModel
from fastapi import HTTPException
from app.core.config import settings


# 1. 配置你的数据库信息 (请替换为你的实际密码和数据库名)
DATABASE_URL = settings.DATABASE_URL

# 2. 创建异步引擎
# echo=True 会在控制台打印生成的 SQL 语句，方便你调试
engine = create_async_engine(
    DATABASE_URL,
    echo=True,#可选，输出SQL日志
    pool_size=10,#设置连接池活跃的连接数量
    max_overflow=20,#允许额外的连接数
    )

# 3. 创建基类
class Base(DeclarativeBase):
    create_time: Mapped[datetime]=mapped_column(
        DateTime,
        insert_default=func.now(),
        default=func.now,
        comment='创建时间')
    update_time: Mapped[datetime]=mapped_column(
        DateTime,
        insert_default=func.now(),
        default=func.now,
        onupdate=func.now(),
        comment='更新时间')

# 4. 创建模型
class Book(Base):
    __tablename__='book'

    id:Mapped[int]=mapped_column(primary_key=True,comment='书籍id')
    bookname:Mapped[str]=mapped_column(String(255),comment='书籍名称')
    author:Mapped[str]=mapped_column(String(255),comment='作者')
    price:Mapped[float]=mapped_column(Float,comment='价格')
    publisher:Mapped[str]=mapped_column(String(255),comment='出版社')

#5.定义函数建表
async def create_table():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 【启动时执行】
    await create_table()
    yield
    # 【关闭时执行】
    await engine.dispose() # 优雅地关闭连接池

app = FastAPI(lifespan=lifespan)

#6.创建会话工厂
AsyncSessionLocal=async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    )

#7.创建获取数据库会话的依赖函数
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception :
            await session.rollback()
            raise
        finally:
            await session.close()

#8.创建获取书籍列表的接口
@app.get("/books/books_list")
async def get_books_list(
    db: AsyncSession = Depends(get_db)
    ):
    #result = await db.execute(select(Book))
    #x= result.scalars().all()#获取所有数据
    #x=result.scalars().first()#获取第一条数据
    x=await db.get(Book,ident=1)#根据id获取书籍
    return x

@app.get("/books/books_list/{id}")
async def get_book(id: int,db: AsyncSession = Depends(get_db)):
    result=await db.execute(select(Book).where(Book.id==id))
    x=result.scalar_one_or_none()
    return x

@app.get("/books/search_book")
async def search_book(
    db: AsyncSession = Depends(get_db)
    ):
    #%表示任意字符，_表示任意一个字符
    #result=await db.execute(select(Book).where(Book.author.like('%K_')))
    id_list=[1,2,3]
    result=await db.execute(select(Book).where(Book.id.in_(id_list)))
    x=result.scalars().all()
    return x

@app.get("/books/count")
async def get_count(db: AsyncSession = Depends(get_db)):
    #result=await db.execute(select(func.count(Book.id)))
    result=await db.execute(select(func.avg(Book.price)))
    x=result.scalar()
    return x

@app.get("/books/get_book_list")
async def get_book_list(
    page: int=2,
    page_size: int=1,
    db: AsyncSession = Depends(get_db)
    ):
    #offset跳过的数量，limit每页的数量
    result=await db.execute(select(Book).offset((page-1)*page_size).limit(page_size))
    x=result.scalars().all()
    return x

class AddBook(BaseModel):
    id: int
    bookname: str
    author: str
    price: float
    publisher: str

@app.post("/books/add_book")
async def add_book(book: AddBook,db: AsyncSession = Depends(get_db)):
    new_book=Book(**book.__dict__)
    db.add(new_book)
    await db.commit()
    return book

class UpdateBook(BaseModel):
    bookname: str
    author: str
    price: float
    publisher: str

@app.put("/books/update_book/{id}")
async def update_book(id: int,book: UpdateBook,db: AsyncSession = Depends(get_db)):
    a=await db.get(Book,id)
    if a is None:
        raise HTTPException(status_code=404,detail="书籍不存在")
    a.bookname=book.bookname
    a.author=book.author
    a.price=book.price
    a.publisher=book.publisher
    await db.commit()
    return book

@app.delete("/books/delete_book/{id}")
async def delete_book(id: int,db: AsyncSession = Depends(get_db)):
    a=await db.get(Book,id)
    if a is None:
        raise HTTPException(status_code=404,detail="书籍不存在")
    await db.delete(a)
    await db.commit()
    return {"message": "书籍删除成功"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)