from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

Base = declarative_base()

class FrameModel(Base):
    __tablename__ = "frames"

    id = Column(Integer, primary_key=True, index=True)
    description = Column(String, index=True)