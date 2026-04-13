from datetime import datetime

from sqlalchemy import Integer, Float, DateTime
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
