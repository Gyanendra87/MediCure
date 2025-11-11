# database.py
import os
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

# SQLite DB URL (default)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./medicure.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ======================
# Models
# ======================

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(128), nullable=False)


class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String(255), nullable=False)
    pdf_text = Column(Text, nullable=True)


# ======================
# Initialize DB
# ======================
def init_db():
    Base.metadata.create_all(bind=engine)
