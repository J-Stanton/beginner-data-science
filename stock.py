import yfinance as yf
import streamlit as st
import pandas as pd

st.write("""
# Simple Stock Price App

Shown are the stock **closing price** and **volume** of google         
""")

tickerSymbol = 'GOOGL'
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(period='1d',start = '2015-01-01', end = '2025-01-01')

st.line_chart(tickerDf.Close)
st.line_chart(tickerDf.Volume)