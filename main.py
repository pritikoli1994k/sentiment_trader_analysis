from src.data_preprocessing import load_sentiment, load_trades, merge_datasets
from src.trader_analysis import per_trader_metrics, daily_performance
from src.correlation_model import prepare_features, train_logistic
from src.visualization import plot_sentiment_over_time, plot_trader_pnl_hist
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def run_pipeline():
    sentiment_path = os.path.join(DATA_DIR, 'fear_greed_index.csv')
    trades_path = os.path.join(DATA_DIR, 'historical_data.csv')
    sentiment = load_sentiment(sentiment_path)
    trades = load_trades(trades_path)
    merged = merge_datasets(trades, sentiment)
    merged.to_csv('results/merged_trades_sentiment.csv', index=False)
    # Metrics
    traders = per_trader_metrics(merged)
    traders.to_csv('results/trader_metrics.csv')
    daily = daily_performance(merged)
    daily.to_csv('results/daily_performance.csv')
    # simple model
    X, y = prepare_features(merged)
    res = train_logistic(X, y)
    print('Model results:', res)
    # plots
    plot_sentiment_over_time(sentiment, outpath='results/sentiment_over_time.png')
    plot_trader_pnl_hist(merged, outpath='results/pnl_hist.png')
    print('Pipeline finished. Check results/ folder.')

if __name__ == '__main__':
    run_pipeline()
