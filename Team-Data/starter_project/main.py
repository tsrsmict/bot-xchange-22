import pickle
import pandas as pd

MODEL_PATH = "model.sav"


"""
Refer to the starter pack document for a complete explanation of the input and output of this function
https://docs.google.com/document/d/1GLSzUIFsBLgRW3jMwuqC1hBKS6GZoRRGJKd6vrlX83o/edit?usp=sharing
"""
def make_trades(
    df,
    holdings,
    current_money,
    day,
):

    # TODO: Your code may use a different number of days
    prices = df.iloc[:, -50:] # last 50 days

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    # TODO: Your model might require a different set of input features 
    preds = model.predict(prices)

    # The last column in the dataframe represents the current price
    current_prices = df.iloc[:, -1]

    # Find the differences between the forecasted prices and the current prices
    diff = preds - current_prices
    diff = pd.Series(diff, dtype=float)

    """
    This is a very simple algorithm for deciding what stock to buy:
    It puts as much money as it can into the stock that is forecasted to rise the most (even if all stocks are predicted to fall in value)
    It then looks at all stocks it currently owns, and sells the stock that is forecasted to fall the most (even if it forecasts all of them to rise!)

    This is an obviously poor algorithm which will probably result in a net loss - make sure to change it!
    If you do not improve on it, it is likely that your model will go to waste.
    """

    # TODO: Rewrite the algorithm below to decide what trades to make

    stock_to_buy = diff.argmax()
    stock_to_buy_price = current_prices[stock_to_buy]
    num_to_buy = current_money // stock_to_buy_price

    current_min = float("inf")
    stock_to_sell = 0
    for stock_name in holdings:
        if holdings[stock_name] == 0:
            # Currently own 0 shares of this stock
            continue

        if diff[stock_name] < current_min:
            current_min = diff[stock_name]
            stock_to_sell = stock_name
    num_to_sell = holdings[stock_to_sell]

    # The lines below are just to demonstrate that you can call code from other files - feel free to remove them
    # TODO: Remember to update this line! The length of your return dictionary will probably be variable and not static, as you may wish to buy or sell a different number of stocks each day.
    return {stock_to_buy: num_to_buy, stock_to_sell: -num_to_sell}
