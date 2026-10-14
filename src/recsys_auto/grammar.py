"""Декларативная грамматика пакетов исторических признаков."""

from dataclasses import dataclass

ENTITIES = ("user", "item", "user_item", "genre", "user_genre")
SCALAR_STATS = ("log_count", "positive_rate", "rating_mean", "gap_days")
MULTI_STATS = (
    "log_count_mean",
    "log_count_max",
    "positive_rate_mean",
    "positive_rate_min",
    "rating_mean",
    "gap_days_min",
)


@dataclass(frozen=True)
class FeaturePack:
    entity: str
    days: int | None

    def __post_init__(self):
        if self.entity not in ENTITIES or (self.days is not None and self.days <= 0):
            raise ValueError("Недопустимая сущность или длина окна")

    @property
    def name(self) -> str:
        window = "all" if self.days is None else f"{self.days}d"
        return f"{self.entity}__{window}"

    @property
    def columns(self) -> list[str]:
        stats = MULTI_STATS if "genre" in self.entity else SCALAR_STATS
        return [f"{self.name}__{stat}" for stat in stats]

    def sql(self) -> str:
        """Иллюстрирует причинное условие и группировку одного пакета."""
        keys = {
            "user": "h.user = q.user",
            "item": "h.item = q.item",
            "user_item": "h.user = q.user AND h.item = q.item",
            "genre": "h.genre = q.genre",
            "user_genre": "h.user = q.user AND h.genre = q.genre",
        }
        lower = (
            "" if self.days is None else f"\n  AND h.timestamp >= q.timestamp - {self.days * 86400}"
        )
        return (
            "SELECT COUNT(*) AS count,\n"
            "       (COALESCE(SUM(h.label), 0) + 2.5) / (COUNT(*) + 5) AS positive_rate,\n"
            "       (COALESCE(SUM(h.rating), 0) + 15) / (COUNT(*) + 5) AS rating_mean\n"
            "FROM history h\nWHERE "
            + keys[self.entity]
            + "\n  AND h.timestamp < q.timestamp"
            + lower
        )


def packs(days: list[int | None]) -> list[FeaturePack]:
    return [FeaturePack(entity, window) for entity in ENTITIES for window in days]
