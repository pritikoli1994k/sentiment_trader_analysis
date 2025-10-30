import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_sentiment_over_time(sentiment_df, outpath=None):
    s = sentiment_df.copy()
    s['Date'] = pd.to_datetime(s['Date'])
    counts = s.groupby(['Date','Classification']).size().unstack(fill_value=0)
    counts.plot(kind='line', figsize=(10,4), title='Fear/Greed counts over time')
    if outpath:
        plt.savefig(outpath, bbox_inches='tight')
    plt.close()

def plot_trader_pnl_hist(trades_df, outpath=None):
    plt.figure(figsize=(8,4))
    sns.histplot(trades_df['closedPnL'].dropna(), kde=False)
    plt.title('Distribution of closedPnL')
    if outpath:
        plt.savefig(outpath, bbox_inches='tight')
    plt.close()
