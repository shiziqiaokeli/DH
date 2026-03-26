import uvicorn
from fastapi import FastAPI  # 导入 FastAPI 框架核心类
from fastapi import Path
from fastapi import Query
from pydantic import BaseModel
from pydantic import Field
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi import HTTPException
from fastapi import Depends

app = FastAPI()  # 创建 FastAPI 应用实例

@app.get("/hello")
async def get_hello():
    return {"msg": "你好FastAPI"}
@app.get("/book/{id}")
async def get_book(id: int=Path(...,gt=0,lt=101,description="书籍id,大于0小于101")):
    id_list=[1,2,3,4,5,6]
    if id not in id_list:
        raise HTTPException(status_code=404,detail="书籍不存在")
    return {"id": id,"title": f"这是第{id}本书"}
@app.get("/author/{name}")
async def get_author(name: str=Path(...,min_length=2,max_length=10,description="作者姓名,长度大于2小于10")):
    return {"msg": f"这是{name}的信息"}
@app.get("/news/news_list")
async def get_news_list(
    skip:int=Query(0,description="跳过的记录数",lt=100),
    limit:int=Query(10,description="返回的记录数")
    ):
    return {"skip": skip,"limit": limit}

class User(BaseModel):
    username: str=Field("张三",min_length=2,max_length=10,description="用户名,长度大于2小于10")
    password: str=Field(min_length=3,max_length=20,description="密码,长度大于3小于20")

@app.post("/register")
async def register(user: User):
    return user

@app.get("/html",response_class=HTMLResponse)
async def get_html():
    return "<h1>Hello, FastAPI!</h1>"
    
@app.get("/file")
async def get_file():
    path="./files/1.jpg"
    return FileResponse(path)

class News(BaseModel):
    id: int
    title: str
    content: str

@app.get("/news/{id}",response_model=News)
async def get_news(id: int):
    return {
        "id": id, 
        "title": f"这是第{id}条新闻", 
        "content": "你好，我是新闻内容"
        }

async def commom_parameters(
    skip:int=Query(0,ge=0),
    limit:int=Query(10,le=60)
):
    return {"skip": skip,"limit": limit}

@app.get("/books/books_list")
async def get_books_list(
    commons:dict=Depends(commom_parameters)
    ):
    return commons
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)