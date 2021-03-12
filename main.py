import pandas as pd
import datetime as dt
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import pickle
import time
from PIL import Image
from SPY_data import *

st.sidebar.title("WallStBets S&P 500 Predictor & Analysis")
option = st.sidebar.radio("Options:", ("Introduction","Ticker Mentions", "Ticker Sentiment", "Stock Lookup", "S&P 500 Analysis"))

if option == "Introduction":
	image = Image.open('wallstbets.jpg')
	st.image(image)
	st.subheader('About WallStBets')
	st.write("r/wallstreetbets, also known as WallStreetBets or WSB, is a subreddit where participants discuss stock and option trading. It has become notable for its profane nature, aggressive trading strategies, and role in the GameStop short squeeze that caused losses on short positions in U.S. firms topping US$70 billion in a few days in early 2021. Our hypothesis is that through analysis of this subreddit, we will be able to find signals that are correlated with a bullish (upward trend) or bearish (downward trend) sentiment of the overall market.")
	st.subheader('How it Works')
	st.write("Our process is this: we scrape top-rated comments from discussion boards on the WallStreetBets, Finance, and Investing subreddits over the past 24 hours and analyze the content for any stock ticker mentions. Once a stock ticker is found, we run the comment through a neural network and vader sentiment analysis is performed. The total number of mentions and the overall sentiment of the stock is kept in memory, alongside all other stocks. This data is then used for a Logistic Regression model to predict if the S&P 500 will go up or down by the end of the day.")

if option == "S&P 500 Analysis":
	st.header('WallStBets S&P 500 Sentiment')
	df = pd.read_csv('data/SPYchart.csv')
	fig = px.line(x=df['Date'], y=df["Normalized SPY"], color=px.Constant("S&P 500"),
             labels=dict(x="Date", y="S&P 500 Sentiment", color="Time Peroid"))
	fig.add_bar(x=df['Date'], y=df['Normalized Sentiment'], name="WallStBets Sentiment")
	fig.update_layout(autosize=False,
		width=1000, height=600,
		margin=dict(l=40, r=40, b=40, t=40))

	st.plotly_chart(fig)

	dfWSB = pd.read_csv('data/dfWSB.csv')

	#get current market data
	dfSPY = dfWSB.loc[dfWSB['Ticker'].isin(SPY)]
	current_day = dfSPY.loc[dfSPY['Date'] == dfSPY['Date'].max()] #gets most recent date
	data = current_day.groupby("Date").aggregate({"Mentions": "sum", "Sentiment": "mean"})

	# load the model from disk
	filename = 'finalized_model.sav'
	loaded_model = pickle.load(open(filename, 'rb'))

	st.subheader('Make Prediction')
	if st.button('Collect S&P Stock Sentiment for Past 24 Hours'):
		st.success('Collecting Data...')
		time.sleep(2)
		st.dataframe(data)
	
	if st.button('Predict Direction of S&P 500'):
		result = loaded_model.predict(data)
		st.success('Logistic Regression Fitting...')
		time.sleep(2)
		if result[0] == 1:
			st.write('Model Prediction: S&P 500 will go UP by EOD.')
		else:
			st.write('Model Prediction: S&P 500 will go DOWN by EOD.')

if option == "Ticker Sentiment":
	st.header('Ticker Sentiment')
	slider = st.slider('Number of Stocks to Analyze', 10, 100, 5)
	df = pd.read_csv('data/tickers.csv')
	df.rename(columns={'Unnamed: 0' : 'Tickers'}, inplace = True)
	filtered_data = df.iloc[:slider]

	st.subheader('Sentiment Of Tickers over the Past 24 Hours')

	fig3 = go.Figure(data=[
	    go.Bar(name='Bearish', x=filtered_data['Tickers'], y=filtered_data['Bearish']),
	    go.Bar(name='Bullish', x=filtered_data['Tickers'], y=filtered_data['Bullish']),
	    go.Bar(name='Neutral', x=filtered_data['Tickers'], y=filtered_data['Neutral']),
	    go.Bar(name='Total/Compound', x=filtered_data['Tickers'], y=filtered_data['Total/Compound']),
	])
	# Change the bar mode
	fig3.update_layout(barmode='group', autosize=False,
		width=1000, height=800,
		margin=dict(l=40, r=40, b=40, t=40))
	st.plotly_chart(fig3)

if option == "Ticker Mentions":
	st.header('Ticker Mentions')
	slider = st.slider('Number of Stocks to Analyze', 10, 100, 5)
	st.subheader('Most Mentioned Tickers Over the Past 24 Hours')
	df = pd.read_csv('data/tickers.csv')
	df.rename(columns={'Unnamed: 0' : 'Tickers'}, inplace = True)
	filtered_data = df.iloc[:slider]

	fig2 = px.treemap(filtered_data, path=['Tickers', 'Mentions'], values= 'Mentions')
	fig2.update_layout(autosize=False,
		width=800, height=700,
		margin=dict(l=40, r=40, b=40, t=40))

	st.plotly_chart(fig2)

if option == "Stock Lookup":
	dfWSB = pd.read_csv('data/dfWSB.csv')

	def find(stock):
		df_stock = dfWSB.loc[dfWSB['Ticker'] == (stock)]
		df_stock.groupby("Date").aggregate({"Mentions": "sum", "Sentiment": "mean"})
		return df_stock

	
	def get_ticker_data(symbol, start_date, end_date):

	    ticker = yf.Ticker(symbol)

	    df_fin = ticker.history(period="1d", start=start_date, end=end_date)
	    df_fin = df_fin[["Close"]]

	    return df_fin

	st.header('Financial Dashboard')
	ticker_input = st.text_input('Please enter your company ticker:', value = 'TSLA')
	if ticker_input not in dfWSB['Ticker'].values:
		st.warning('Stock Symbol not found. Please try again.')
		st.stop()
	st.success('Analyzing stock...')

	search_button = st.button('Search')

	df_fin = get_ticker_data(ticker_input, start_date = "2018-08-01", end_date =  dt.date.today())
	df_stock = find(ticker_input)

	fig4 = px.line(x=df_fin.index, y=df_fin["Close"],
		labels={'x': '','y': f'{ticker_input} Share Price'})
	fig5 = px.bar(x=df_stock['Date'], y=df_stock['Sentiment'], 
		labels ={'y' : 'Sentiment'}, color_discrete_sequence=["red"])
	fig6 = px.bar(x=df_stock['Date'], y=df_stock['Mentions'],
	labels ={'x' : 'Date', 'y' : 'Mentions'}, color_discrete_sequence=["purple"])

	fig4.update_layout(autosize=False,
		width=1000, height=500,
		margin=dict(l=40, r=40, b=40, t=40))
	fig4.update_xaxes(visible=False, showticklabels=True)

	fig5.update_layout(autosize=False,
		width=1000, height=200,
		margin=dict(l=40, r=40, b=40, t=40))
	fig5.update_xaxes(visible=False, showticklabels=True)
	fig6.update_layout(autosize=False,
		width=1000, height=200,
		margin=dict(l=40, r=40, b=40, t=40))
	fig6.update_xaxes(visible=True, showticklabels=True)

	st.plotly_chart(fig4)
	st.plotly_chart(fig5)
	st.plotly_chart(fig6)