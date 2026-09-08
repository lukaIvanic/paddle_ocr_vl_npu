"""Three CLI choices; region identity is independent of recognizer choice."""
from dataclasses import dataclass

TASKS = {"OCR:": "text", "Table Recognition:": "table", "Formula Recognition:": "formula"}


@dataclass(frozen=True)
class Routing:
    text: str = "unirec"
    table: str = "paddle"
    formula: str = "paddle"

    def model_for(self, prompt):
        return getattr(self, TASKS[prompt])

    @property
    def models(self):
        return tuple(name for name in ("unirec", "paddle") if name in (self.text, self.table, self.formula))


def add_arguments(parser):
    for task, default in (("text", "unirec"), ("table", "paddle"), ("formula", "paddle")):
        parser.add_argument(f"--{task}-model", choices=("unirec", "paddle"), default=default)

