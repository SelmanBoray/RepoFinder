"""Core engine of the sample package."""
from __future__ import annotations

import os
import os.path as osp, sys
from typing import Optional

from mypkg.sub import models
from mypkg.sub.models import User as Account, Base
from .sub.models import *


def run_engine(
    engine: "Engine",
    *,
    retries: int = 3,  # how many times to retry
) -> bool:
    """Run the engine once.

    Returns True on success.
    """

    def attempt(n):
        """Single attempt."""
        return n > 0

    class Result:
        ok = True

    return attempt(retries)


class Engine(Base):
    # A comment before the docstring is not a statement.
    """An engine that does work."""

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def label(self) -> str:
        return self.name.upper()

    @label.setter
    def label(self, value: str) -> None:
        self.name = value

    @classmethod
    @staticmethod
    def build(cls):
        return cls("default")

    async def start(self, timeout: Optional[float] = None):
        '''Start asynchronously.'''
        def on_done():
            def deeper():
                pass
            return deeper
        return on_done

    if sys.platform == "win32":
        def platform_hook(self):
            return "windows"
