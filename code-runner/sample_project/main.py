import warnings

warnings.filterwarnings("ignore")
import pickle
import pandas as pd


with open("model", "rb") as file:
    model = pickle.load(file)

log_data = pd.DataFrame()

def predict(
    df,
    holdings,
    current_money,
    day,
):
    day = day
    new = pd.DataFrame()
    new["std"] = df.std(axis=1)
    new["min"] = df.min(axis=1)
    new["max"] = df.max(axis=1)
    new["mean"] = df.mean(axis=1)
    new.reset_index(inplace=True)

    prices = df.iloc[0:, 100:149]
    new = pd.concat([new, prices], axis=1)

    new.drop(["index"], axis=1, inplace=True)

    preds = model.predict(new)
    log_data[day] = preds
    print(hash(str(preds)))

    current_prices = df.iloc[:, -1]
    diff = preds - current_prices
    diff = pd.Series(diff, dtype=float)

    transactions = {}
    stocks_to_buy = diff[diff > 0]  # type: ignore
    stocks_to_buy /= stocks_to_buy.sum()
    for stock, frac in stocks_to_buy.items():
        num = frac * current_money / current_prices[stock]
        transactions[stock] = int(num)

    stocks_to_sell = diff[diff < 0]  # type: ignore
    for stock in stocks_to_sell[
        stocks_to_sell.index.map(lambda i: holdings[i] > 0)
    ].index:
        transactions[stock] = -holdings[stock]

    return transactions
