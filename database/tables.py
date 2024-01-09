import os

from sqlalchemy import create_engine, Column, Integer, DateTime, String, Float, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()


DATABASE_DIR = os.path.dirname(__file__)


class Pixel(Base):
    __tablename__ = 'pixel'
    id = Column(Integer, primary_key=True)
    longitude = Column(Float)
    latitude = Column(Float)
    value = Column(Float)

    __table_args__ = (
        Index('ix_pixel_longitude', 'longitude'),
        Index('ix_pixel_latitude', 'latitude'),
    )
