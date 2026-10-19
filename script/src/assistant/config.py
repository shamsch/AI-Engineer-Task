import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

DATA_DIR: Path = Path(os.getenv("DATA_DIR", Path(__file__).parents[3] / "data")).resolve()
MODEL: str = os.getenv("MODEL", "openai:gpt-5-mini")
