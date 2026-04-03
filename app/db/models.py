from typing import Optional
from sqlalchemy import String, Text, ForeignKey, Integer, Boolean, Float
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

# 1. 知识库映射表
class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), unique=True, comment="用户自定义名称")
    collection_name: Mapped[str] = mapped_column(String(100), comment="向量数据库中的Collection名")

# 2. 提示词表
class Prompt(Base):
    __tablename__ = "prompts"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), unique=True, comment="用户自定义名称")
    content: Mapped[str] = mapped_column(Text, comment="系统提示词正文")

# 3. 语音模型表
class VoiceModel(Base):
    __tablename__ = "voice_models"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), unique=True, comment="用户自定义名称")
    ckpt_path: Mapped[str] = mapped_column(String(255), comment="模型权重路径")
    ref_audio_path: Mapped[str] = mapped_column(String(255), comment="参考音频路径")

# 4. 全局活跃配置表 (单例模式)
class SystemSetting(Base):
    __tablename__ = "system_settings"
    
    # 我们固定 id=1，确保全系统只有一套活跃配置
    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    
    # 关联外键：直接绑定到上面三张表的主键
    active_kb_id: Mapped[int] = mapped_column(ForeignKey("knowledge_bases.id"))
    active_prompt_id: Mapped[int] = mapped_column(ForeignKey("prompts.id"))
    active_model_id: Mapped[int] = mapped_column(ForeignKey("voice_models.id"))
    t_value: Mapped[float] = mapped_column(Float, default=0.1, comment="温度参数")
    is_voice_mode: Mapped[bool] = mapped_column(Boolean, default=False, comment="是否为语音模式: True(语音)/False(文本)")