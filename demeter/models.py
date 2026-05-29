from datetime import datetime

from sqlalchemy import Integer, Float, DateTime, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SolarState(Base):
    """Single-row coulomb counter state. id is always 1."""
    __tablename__ = "solar_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    current_wh: Mapped[float] = mapped_column(Float, nullable=False)
    soc_percent: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class DecisionLog(Base):
    __tablename__ = "decision_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    observation_json: Mapped[str] = mapped_column(Text, nullable=False)
    action_json: Mapped[str] = mapped_column(Text, nullable=False)
    policy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str] = mapped_column(String(128), nullable=False)
    reward: Mapped[float | None] = mapped_column(Float, nullable=True)
