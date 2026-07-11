from __future__ import annotations

from datetime import datetime
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = "sqlite:///./data/app.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    profile = relationship("Profile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    scans = relationship("ScanHistory", back_populates="user", cascade="all, delete-orphan")


class Profile(Base):
    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    chronic_conditions = Column(String, default="")  # comma-separated
    allergens = Column(String, default="")  # comma-separated
    daily_calorie_target_kcal = Column(Float, nullable=True)
    daily_protein_target_g = Column(Float, nullable=True)
    daily_fat_target_g = Column(Float, nullable=True)
    daily_carbohydrate_target_g = Column(Float, nullable=True)
    objective = Column(String, default="alerji_takibi")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="profile")


class ScanHistory(Base):
    __tablename__ = "scan_histories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    product_id = Column(String, nullable=False)
    category = Column(String, nullable=False)
    category_confidence = Column(Float, nullable=False)
    nutrition_json = Column(String, nullable=False)  # JSON-serialized
    detected_allergens = Column(String, default="")  # comma-separated
    risk_flags = Column(String, default="")  # comma-separated
    ocr_confidence = Column(Float, nullable=False)
    file_hash = Column(String, nullable=True, index=True)
    explanation_text = Column(String, nullable=True)
    scanned_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="scans")


def init_db() -> None:
    os.makedirs("./data", exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
