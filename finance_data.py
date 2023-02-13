import yfinance as yf


# Transforming Data for Analysis
import yfinance as yf
start_date = '2015-04-01'
end_date = '2021-04-01'
ticker = 'GOOGL'
data = yf.download(ticker, start_date, end_date)
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)
print(data.tail())
print(data.head())

# Export data to a CSV file
data.to_csv("images/GOOGL.csv")
