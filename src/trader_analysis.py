import pandas as pd
import numpy as np

def per_trader_metrics(trades_df):
    # computes basic metrics per account
    g = trades_df.groupby('account').agg(
        trades_count=('account','size'),
        total_pnl=('closedPnL','sum'),
        avg_leverage=('leverage','mean'),
        win_rate=('closedPnL', lambda x: (x>0).mean())
    )
    return g.sort_values('total_pnl', ascending=False)

def daily_performance(trades_df):
    return trades_df.groupby('date').agg(
        daily_pnl=('closedPnL','sum'),
        avg_leverage=('leverage','mean'),
        trades_count=('account','count')
    ).reset_index()
