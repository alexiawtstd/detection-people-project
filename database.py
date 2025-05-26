from sqlalchemy import Column, Integer, String, DateTime, Float, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

class MediaFile(Base):
    __tablename__ = 'media_files'

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    original_filepath = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    upload_time = Column(DateTime, default=datetime.now)
    processing_time = Column(Float)
    duration = Column(Float)
    frame_count = Column(Integer)
    people_count = Column(Integer)
    file_size = Column(Integer)
    status = Column(String, default='processed')


engine = create_engine('sqlite:///media_analysis.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
