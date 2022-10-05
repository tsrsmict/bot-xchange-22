import pandas as pd
import main

# For the purpose of optimising your algorithms you can use the training data but the stocks we will use to test your code will be completely different.
df = pd.read_csv("../training_data.csv") 
df = df[:25]
TOTAL_DAYS = 50
INITIAL_MONEY = 10_000_000

current_money = INITIAL_MONEY
holdings = {i: 0 for i in range(25)}

# Iterate for remaining 50 days
for day in range(TOTAL_DAYS):
    day_index = day - TOTAL_DAYS # -50, -49, ..., -1
    data = df.iloc[:, :day_index]
    transactions = main.make_trades(data, holdings, current_money, day)

    current_prices = df.iloc[:, day_index - 1]
    for stock, num_shares in transactions.items():
        price = current_prices[stock]
        # The following code is correct for both +ve and -ve `num_shares`
        amount = num_shares * price
        holdings[stock] += num_shares
        current_money -= amount

    print(f"{day=} {current_money=}")

# Sell all remaining stock in holdings
current_prices = df.iloc[:, -1]
for stock, num_shares in holdings.items():
    price = current_prices[stock]
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
