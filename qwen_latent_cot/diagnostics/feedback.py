"""Checked feedback label schema for C2 data preparation."""
from dataclasses import dataclass
import json

@dataclass(frozen=True)
class Feedback:
    observation: str
    gap: str
    correction: str
    satisfied: bool = False
    origin: str = 'manual_checked'
    raw: str | None = None

    def __post_init__(self):
        if any(not isinstance(value, str) for value in (self.observation, self.gap, self.correction)):
            raise ValueError('feedback observation/gap/correction must be strings')
        if type(self.satisfied) is not bool:
            raise ValueError('feedback satisfied must be a boolean')
        if not self.satisfied and not self.correction.strip():
            raise ValueError('an unsatisfied preview requires a positive correction')
        if self.origin not in ('manual_checked', 'bagel_auto', 'source_annotation'):
            raise ValueError('unknown feedback provenance')

    @classmethod
    def parse(cls, raw):
        text = raw.strip()
        if text.startswith('```') and text.endswith('```'):
            text = '\n'.join(text.splitlines()[1:-1])
        value = json.loads(text)
        if set(value) != {'observation', 'gap', 'correction', 'satisfied'}:
            raise ValueError('reflection must return the four declared JSON fields')
        return cls(**value, origin='bagel_auto', raw=raw)
