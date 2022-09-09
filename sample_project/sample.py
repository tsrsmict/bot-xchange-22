import pickle
import pandas as pd
from sklearn import metrics

from typing import Mapping


df: pd.DataFrame = pd.read_csv("./opening_prices_biotech_complete.csv")  # type: ignore


stock_name_t = int
individual_prices_t = Mapping[str, float]
current_prices_t = Mapping[stock_name_t, individual_prices_t]

holdings_t = Mapping[stock_name_t, int]

return_t = Mapping[stock_name_t, int]

market_data_dict: current_prices_t = df.to_dict("index")  # type:ignore


def predict(
    input_dict: current_prices_t,
    holdings: holdings_t = {},
    current_money: float = 1000,
    day: int = 0,
) -> return_t:
    df = pd.DataFrame(input_dict).transpose()
    df.drop(["ticker", "Unnamed: 0"], axis=1, inplace=True)

    new = pd.DataFrame()
    new["std"] = df.std(axis=1)
    new["min"] = df.min(axis=1)
    new["max"] = df.max(axis=1)
    new["mean"] = df.mean(axis=1)
    new.reset_index(inplace=True)

    prices = df[df.columns[100:149]]
    new = pd.concat([new, prices], axis=1)

    new.drop(["index"], axis=1, inplace=True)
    new.fillna(0, inplace=True)

    model = pickle.load(open("model.sav", "rb"))
    preds = model.predict(new)

    current_prices = df[df.columns[-1]]
    diff: pd.Series = preds - current_prices
    diff = pd.Series(diff, dtype=float)

    stock_to_buy: stock_name_t = diff.argmax()
    stock_to_buy_price = current_prices[stock_to_buy]
    num_to_buy = current_money // stock_to_buy_price

    current_min = float("inf")
    stock_to_sell = 0
    for stock_name in holdings:
        if holdings[stock_name] == 0:
            continue
        if diff[stock_name] < current_min:
            current_min = diff[stock_name]
            stock_to_sell = stock_name
    num_to_sell = holdings[stock_to_sell]

    return {stock_to_buy: num_to_buy, stock_to_sell: -num_to_sell}

holdings = {i:0 for i in range(0, 48)}
current_money = 1000
for day in range(1):
    ret = predict(market_data_dict, holdings, current_money, day)
    for stock, amount in ret.items():
        holdings[stock] += amount
    breakpoint()
# true = df[df.columns[149]]
#
# print("Mean Absolute Error:", metrics.mean_absolute_error(true, preds))
