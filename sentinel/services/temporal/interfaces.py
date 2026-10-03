from typing import Protocol

from services.temporal.types import CandidateSignal, TemporalContext


class TemporalAnalyzer(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def analyze(self, context: TemporalContext) -> list[CandidateSignal]: ...


class NullAnalyzer:
    name = "null-analyzer"
    version = "0.0.0"

    def analyze(self, context: TemporalContext) -> list[CandidateSignal]:
        return []
