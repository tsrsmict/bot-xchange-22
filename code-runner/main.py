PROJECT_NAME = "sample_project"
TOTAL_DAYS = 50
INITIAL_MONEY = 2000
print=lambda *_,**__:None

import importlib
import os
import sys
from typing import Callable, Mapping

import pandas as pd

baseDir = os.path.dirname(__file__)
os.chdir(baseDir + "/" + PROJECT_NAME)
sys.path.append(".")

stock_name_t = int
current_prices_t = pd.DataFrame  # index: stock_name_t, columns: str, values: money_t
holdings_t = Mapping[stock_name_t, int]
transactions_t = Mapping[stock_name_t, int]

func_t = Callable[[current_prices_t, holdings_t, float, int], transactions_t]

predictor: func_t = importlib.import_module(PROJECT_NAME + ".main").predict

df = pd.read_csv(baseDir + "/opening_prices_biotech_complete.csv")
df.drop(["ticker", "Unnamed: 0"], axis=1, inplace=True)
num_of_stocks = len(df)
current_money = INITIAL_MONEY
holdings = {i: 0 for i in range(num_of_stocks)}

for day in range(TOTAL_DAYS):
    print(f"Day {day}:")
    day_index = day - TOTAL_DAYS
    data = df.iloc[:, :day_index]
    today_data = df.iloc[:, day_index - 1]
    try:
        transactions = predictor(data, holdings, current_money, day)
    except Exception as e:
        print("Error occured: ", e)
        continue
    for stock, num_shares in transactions.items():
        price = today_data[stock]
        amount = num_shares * price
        if num_shares == 0:
            continue
        elif num_shares > 0:
            if amount > current_money:
                print(
                    f"Tried to buy {stock=} {price=} {amount=} {num_shares=} {current_money=}"
                )
                continue
            print(f"Bought {num_shares} of stock {stock} at price {price}")
        else:
            if abs(num_shares) > holdings[stock]:
                print(
                    f"Tried to sell {stock=} {price=} {amount=} {num_shares=} {holdings[stock]=}"
                )
                continue
            print(f"Sold {num_shares} of stock {stock} at price {price}")
        holdings[stock] += num_shares
        current_money -= amount
        assert all(
            holdings[stock] >= 0 for stock in range(num_of_stocks)
        ), "Negative holding not allowed"
        assert current_money >= 0, "Negative money not allowed"
    print(f"Money at end of day {current_money}")
    # print(f"{day=} {current_money=}")

today_data = df.iloc[:, -1]
for stock, num_shares in holdings.items():
    price = today_data[stock]
    amount = num_shares * price
    current_money += amount

delta = current_money - INITIAL_MONEY
print(
    "Final money:",
    current_money,
    "with increase of",
    round(delta / INITIAL_MONEY * 100, 2),
    "\b%",
)
breakpoint()
