import pickle
import pandas as pd

def predict(
    input_dict,
    holdings,
    current_money,
    day,
):
    day = day
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

    with open("model", "rb") as file:
        model = pickle.load(file)
    preds = model.predict(new)

    current_prices = df[df.columns[-1]]
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

    return {stock_to_buy: num_to_buy, stock_to_sell: -num_to_sell}
