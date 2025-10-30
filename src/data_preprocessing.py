import pandas as pd

def load_sentiment(path):
    df = pd.read_csv(path, parse_dates=['Date'], dayfirst=False)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    # Expect columns: Date, Classification (Fear/Greed)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df['Classification'] = df['Classification'].str.strip().str.title()
    return df

def load_trades(path):
    df = pd.read_csv(path, parse_dates=['time','start position'], low_memory=False)
    # normalize time to date
    df['date'] = pd.to_datetime(df['time']).dt.date
    return df

def merge_datasets(trades, sentiment):
    # Merge on date
    merged = trades.merge(sentiment, left_on='date', right_on='Date', how='left')
    return merged

if __name__ == '__main__':
    print('data_preprocessing module: provide functions for loading and merging CSVs.')
