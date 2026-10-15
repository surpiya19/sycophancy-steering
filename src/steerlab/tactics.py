"""Surface-level detection of pressure tactics in a prompt (regex, no model).

Patterns live in `resources/tactics.yaml`, or any YAML file mapping a tactic name
to a list of regexes, so they can be extended without touching code. This is a
lexical heuristic: expect false positives ("what if", "sad") and misses
(paraphrases). The model-based reading is reported separately, never folded in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml


@dataclass
class TacticReport:
    hits: dict[str, list[str]]
    severity: int
    level: str

    @property
    def n_tactics(self) -> int:
        return len(self.hits)


class TacticAuditor:
    def __init__(self, patterns_path: str | Path | None = None, points_per_tactic: int = 25):
        if patterns_path is None:
            raw = resources.files("steerlab").joinpath("resources/tactics.yaml").read_text(encoding="utf-8")
        else:
            raw = Path(patterns_path).read_text(encoding="utf-8")
        spec = yaml.safe_load(raw)
        self.patterns = {t: [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in pats] for t, pats in spec.items()}
        self.points = points_per_tactic

    def detect(self, text: str) -> dict[str, list[str]]:
        hits = {}
        for tactic, regexes in self.patterns.items():
            found = [m.group(0) for r in regexes for m in r.finditer(text)]
            if found:
                hits[tactic] = list(dict.fromkeys(found))
        return hits

    def audit(self, text: str) -> TacticReport:
        hits = self.detect(text)
        severity = min(len(hits) * self.points, 100)
        level = "high" if severity >= 60 else "moderate" if severity >= 25 else "low"
        return TacticReport(hits, severity, level)
