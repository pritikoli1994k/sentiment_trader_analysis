import pandas as pd

def sentiment_summary(sentiment_df):
    # counts by day and classification
    return sentiment_df.groupby(['Date','Classification']).size().unstack(fill_value=0)

def map_sentiment_to_numeric(sentiment_df):
    mapping = {'Fear': -1, 'Greed': 1}
    s = sentiment_df.copy()
    s['sentiment_score'] = s['Classification'].map(mapping).fillna(0)
    return s
