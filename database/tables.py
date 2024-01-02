import os

from sqlalchemy import create_engine, Column, Integer, DateTime, String, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()


DATABASE_DIR = os.path.dirname(__file__)


class File(Base):
    __tablename__ = 'file'
    id = Column(Integer, primary_key=True)

    timestamp = Column(DateTime)
    name = Column(String, unique=True)

    min_lon = Column(Float)
    min_lat = Column(Float)
    max_lon = Column(Float)
    max_lat = Column(Float)

    lon_res = Column(Float)
    lat_res = Column(Float)

    # relationships

    # many to one
    pixels = relationship('Pixel', back_populates='file', cascade='delete, delete-orphan')


class Pixel(Base):
    __tablename__ = 'pixel'
    id = Column(Integer, primary_key=True)

    longitude = Column(Float)
    latitude = Column(Float)

    value = Column(Float)

    # relationships

    # one to many
    _file_id = Column(Integer, ForeignKey('file.id'), nullable=True)
    file = relationship('File', back_populates='pixels')


engine = create_engine(f'sqlite:///{DATABASE_DIR}/database.db')
Base.metadata.create_all(engine)
