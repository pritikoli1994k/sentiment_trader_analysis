# import streamlit as st
# import pandas as pd
# import os
# from src.data_preprocessing import load_sentiment, load_trades, merge_datasets
# from src.trader_analysis import per_trader_metrics, daily_performance
# from src.correlation_model import prepare_features, train_logistic
# from src.visualization import plot_sentiment_over_time, plot_trader_pnl_hist
# from io import BytesIO

# st.set_page_config(page_title='Sentiment vs Traders', layout='wide')

# st.title('Bitcoin Market Sentiment vs Trader Performance')

# uploaded_sent = st.file_uploader('Upload fear_greed_index.csv', type='csv', key='s')
# uploaded_trades = st.file_uploader('Upload historical_data.csv', type='csv', key='t')

# if uploaded_sent and uploaded_trades:
#     sentiment = load_sentiment(uploaded_sent)
#     trades = load_trades(uploaded_trades)
#     merged = merge_datasets(trades, sentiment)
#     st.subheader('Merged sample')
#     st.dataframe(merged.head())
#     st.subheader('Trader Metrics (top 10)')
#     st.dataframe(per_trader_metrics(merged).head(10))
#     if st.button('Run simple model'):
#         X,y = prepare_features(merged)
#         res = train_logistic(X,y)
#         st.write('Model accuracy:', res['accuracy'])
#         st.write('ROC AUC:', res['roc_auc'])
# else:
#     st.info('Upload both CSV files to enable analysis.')

# import streamlit as st
# import pandas as pd
# from src.correlation_model import train_logistic

# st.title("📊 Sentiment & Trader Analysis Dashboard")

# uploaded = st.file_uploader("Upload your cleaned CSV", type=['csv'])

# if uploaded:
#     df = pd.read_csv(uploaded)
#     st.write("✅ Data Preview:")
#     st.dataframe(df.head())

#     if 'target' not in df.columns:
#         st.error("Your CSV must include a 'target' column for model training. Please upload a valid file.")
#         st.stop()
#     else:
#             X = df  # keep all columns
#             res = train_logistic(X, y)

#             if "accuracy" in res:
#                 st.success(f"✅ Model Accuracy: {res['accuracy']:.2f}")
#                 st.json(res["report"])
#             else:
#                 st.error(f"❌ Error: {res.get('error', 'Unknown error occurred.')}")


import streamlit as st
import pandas as pd

st.title("📊 Sentiment & Trader Analysis Dashboard")

# Upload CSV file
uploaded = st.file_uploader("Upload your cleaned CSV", type=['csv'])

if uploaded:
    # Read and display CSV
    df = pd.read_csv(uploaded)
    st.success("✅ File uploaded successfully!")
    st.subheader("📋 Data Preview:")
    st.dataframe(df.head())

    # Display basic info
    st.subheader("📈 Dataset Information:")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write("Column names:", list(df.columns))

    # Optional: basic statistics
    if st.checkbox("Show summary statistics"):
        st.write(df.describe())

    # Optional: data type info
    if st.checkbox("Show column data types"):
        st.write(df.dtypes)

else:
    st.info("📂 Please upload a CSV file to begin.")
