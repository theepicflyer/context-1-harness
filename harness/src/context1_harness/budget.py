"""Token budget tracking for the harness agent loop."""

from dataclasses import dataclass


@dataclass
class Budget:
    limit: int = 32768
    warn_frac: float = 0.60
    hard_frac: float = 0.95
    used: int = 0

    def update(self, input_tokens: int, output_tokens: int) -> None:
        """Set used tokens from the latest API call's totals (current context size, not cumulative)."""
        self.used = input_tokens + output_tokens

    @property
    def pct(self) -> float:
        return self.used / self.limit if self.limit else 0.0

    @property
    def level(self) -> str:
        if self.pct >= self.hard_frac:
            return "hard"
        if self.pct >= self.warn_frac:
            return "warn"
        return "ok"

    def status_line(self) -> str:
        pct = round(self.pct * 100)
        line = f"[context: {self.used:,}/{self.limit:,} tokens — {pct}%]"
        level = self.level
        if level == "hard":
            line += (
                f" WARNING: context is above {round(self.hard_frac * 100)}% —"
                " only prune_chunks and complete are allowed."
            )
        elif level == "warn":
            line += (
                f" WARNING: context is above {round(self.warn_frac * 100)}% —"
                " prune irrelevant chunks with prune_chunks or finish with complete."
            )
        return line
