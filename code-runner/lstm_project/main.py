import keras.models
import pandas as pd


model = keras.models.load_model("./lstm_model")
def make_trades(
    df,
    holdings,
    current_money,
    day,
):
    day = day

    num = len(df)
    mat = df.iloc[:,-50:].values
    mat = mat.reshape((num, 50, 1))

    preds_arr = model.predict(mat)  # type: ignore
    preds_arr = preds_arr.reshape((num,))

    preds = pd.Series(preds_arr)

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

    return {stock_to_buy: num_to_buy, stock_to_sell: -num_to_sell}
