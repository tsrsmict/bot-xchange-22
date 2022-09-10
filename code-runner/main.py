import warnings
warnings.filterwarnings("ignore")

import pandas as pd
df = pd.read_csv("./opening_prices_biotech_complete.csv")

import sys
import os
sys.path.append(r".")
os.chdir(r"./the_project")

from the_project import main

from typing import Mapping, Callable

TOTAL_DAYS = 50

stock_name_t = int
individual_prices_t = Mapping[str, float]
current_prices_t = Mapping[stock_name_t, individual_prices_t]

holdings_t = Mapping[stock_name_t, int]

return_t = Mapping[stock_name_t, int]

market_data_dict = df.to_dict("index")

func_t = Callable[[current_prices_t, holdings_t, float, int], return_t]

holdings = {i: 0 for i in range(0, 48)}
current_money = 1000
for day in range(TOTAL_DAYS):
    data = df[df.columns[: day - TOTAL_DAYS]]
    data_dict = data.to_dict("index")  # type:ignore
    ret = main.predict(data_dict, holdings, current_money, day)
    today_data = df[df.columns[day - TOTAL_DAYS - 1]]
    for stock, num_shares in ret.items():
        price = today_data[stock]
        amount = num_shares * price
        if num_shares > 0:
            if amount > current_money:
                print(
                    f"Tried to buy {stock=} {price=} {amount=} {num_shares=} {current_money=}"
                )
                continue
        else:
            if num_shares > holdings[stock]:
                print(
                    f"Tried to sell {stock=} {price=} {amount=} {num_shares=} {holdings[stock]=}"
                )
                continue
        holdings[stock] += num_shares
        current_money -= amount
    print(f"{day=} {current_money=}")

today_data = df[df.columns[-1]]
for stock, num_shares in holdings.items():
    price = today_data[stock]
    amount = num_shares * price
    current_money += amount

print(current_money)
