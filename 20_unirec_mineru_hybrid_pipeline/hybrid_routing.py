"""Experiment 20 routes; the experiment 18 coordinator remains unchanged."""
from dataclasses import dataclass

TASKS = {"OCR:": "text", "Table Recognition:": "table", "Formula Recognition:": "formula"}
MINERU_TASKS = {"OCR:": "text", "Table Recognition:": "table", "Formula Recognition:": "equation"}


def add_arguments(parser):
    for task, default in (("text", "unirec"), ("table", "mineru"), ("formula", "mineru")):
        parser.add_argument(f"--{task}-model", choices=("unirec", "mineru"), default=default)


@dataclass(frozen=True)
class Routing:
    text: str = "unirec"
    table: str = "mineru"
    formula: str = "mineru"

    def model_for(self, prompt):
        return getattr(self, TASKS[prompt])

    @property
    def models(self):
        return tuple(name for name in ("unirec", "mineru")
                     if name in (self.text, self.table, self.formula))
