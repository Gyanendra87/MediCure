import os
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, Date, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from dotenv import load_dotenv
from datetime import datetime

# ==============================
# Load environment variables
# ==============================
load_dotenv()

# ==============================
# Database Config
# ==============================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./medicure.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==============================
# Models
# ==============================

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


class Doctor(Base):
    __tablename__ = "doctors"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    specialization = Column(String(100), nullable=False)
    city = Column(String(100), nullable=False)
    bio = Column(Text, nullable=True)
    contact = Column(String(100), nullable=True)

    # ✅ Replaced 'experience' and 'max_daily_appointments' with 'available_slots'
    available_slots = Column(Integer, default=5)

    appointments = relationship("Appointment", back_populates="doctor")


class Appointment(Base):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False)
    patient_name = Column(String(100), nullable=False)
    patient_city = Column(String(100), nullable=True)
    date = Column(Date, default=datetime.utcnow)
    status = Column(String(50), default="Booked")

    doctor = relationship("Doctor", back_populates="appointments")

# ==============================
# Initialize Database
# ==============================
def init_db():
    Base.metadata.create_all(bind=engine)
