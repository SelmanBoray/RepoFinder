"""Data models."""
import json
from dataclasses import dataclass

from .. import core
from ..core import run_engine as run


class Base:
    """Base model."""

    def to_json(self) -> str:
        return json.dumps(self.__dict__)


@dataclass
class User(Base):
    """A user."""

    name: str
    email: str = ""

    class Meta:
        table = "users"

        def describe(cls):
            return cls.table
