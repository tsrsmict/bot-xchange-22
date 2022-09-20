from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass
class DayData:
    prices: pd.Series
    money: float = 0
    asset_value: float = 0
    bought: dict[int, int] = field(default_factory=dict)
    sold: dict[int, int] = field(default_factory=dict)
    holdings: dict[int, int] = field(default_factory=dict)
    exceptions: Any = field(default_factory=list)

    def calculate_assets(self):
        self.asset_value = 0
        for stock, holding in self.holdings.items():
            self.asset_value += self.prices[stock] * holding


class History:
    daily_data: list[DayData]

    def __init__(self, name="Unnamed"):
        self.name = name
        self.daily_data = []
