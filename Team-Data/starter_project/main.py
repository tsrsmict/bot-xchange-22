import pickle
import pandas as pd

from src import helper1, helper2

def make_trades(
    df,
    holdings,
    current_money,
    day,
):
    prices = df.iloc[:,100:149] # last 50 days

    with open("model", "rb") as file:
        model = pickle.load(file)

    preds = model.predict([day, prices])

    current_prices = df.iloc[:, -1]
    diff = preds - current_prices
    diff = pd.Series(diff, dtype=float)

    stock_to_buy = diff.argmax()
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

    helper1.helper1()
    helper2.helper2()

    return {stock_to_buy: num_to_buy, stock_to_sell: -num_to_sell}
