from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime

Base = declarative_base()


class MediaFile(Base):
    __tablename__ = 'media_files'

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    original_filepath = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    upload_time = Column(DateTime, default=datetime.datetime.utcnow)
    processing_time = Column(Float)
    width = Column(Integer)
    height = Column(Integer)
    duration = Column(Float)
    frame_count = Column(Integer)
    people_count = Column(Integer)
    status = Column(String, default='processed')
    analytics = relationship("VideoAnalytics", back_populates="media", uselist=False)  # ← ДОБАВЬ ЭТО
    # если не используешь сейчас — эти можно убрать:
    # frames = relationship("VideoFrame", back_populates="media")
    # movements = relationship("PersonMovement", back_populates="media")


class VideoAnalytics(Base):
    __tablename__ = 'video_analytics'

    id = Column(Integer, primary_key=True)
    media_id = Column(Integer, ForeignKey('media_files.id'), unique=True)
    duration = Column(Float, nullable=False)
    frame_count = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    fps = Column(Float, nullable=False)
    video_format = Column(String, name='format')
    file_size = Column(Integer)
    processing_time = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    media = relationship("MediaFile", back_populates="analytics")  # ← ЭТО ОБЯЗАТЕЛЬНО


# class DetectedPerson(Base):
#     __tablename__ = 'detected_people'
#
#     id = Column(Integer, primary_key=True)
#     frame_id = Column(Integer, ForeignKey('video_frames.id'))
#     person_id = Column(Integer, nullable=False)
#     x1 = Column(Float, nullable=False)
#     y1 = Column(Float, nullable=False)
#     x2 = Column(Float, nullable=False)
#     y2 = Column(Float, nullable=False)
#     confidence = Column(Float, nullable=False)
#     is_first_frame = Column(Boolean)
#     is_last_frame = Column(Boolean)
#
#     frame = relationship("VideoFrame", back_populates="people")


# Инициализация базы данных
engine = create_engine('sqlite:///media_analysis.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
