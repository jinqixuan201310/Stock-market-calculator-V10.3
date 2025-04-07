# V10.3 Stock Predictor - Streamlit App Version (Improved Stability)
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import streamlit as st

# --- CONFIG ---
NEWS_COUNT = 5

def get_news_headlines(ticker):
    try:
        url = f"https://www.google.com/search?q={ticker}+stock+news&tbm=nws"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('div', class_='BNeawe vvjwJb AP7Wnd')
        return [r.get_text() for r in results[:NEWS_COUNT]]
    except:
        return []

def get_sentiment_score(headlines):
    if not headlines:
        return 0  # Neutral if no news available
    analyzer = SentimentIntensityAnalyzer()
    total_score = 0
    for headline in headlines:
        sentiment = analyzer.polarity_scores(headline)
        total_score += sentiment['compound']
    avg_score = total_score / len(headlines)
    if avg_score > 0.25:
        return 2
    elif avg_score < -0.25:
        return -2
    else:
        return 0

def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="7d")
        volume_spike = hist['Volume'].iloc[-1] > 1.3 * hist['Volume'].mean() if not hist.empty else False

        prices = yf.download(ticker, period="21d", progress=False)['Close']
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        latest_rsi = rsi.dropna().iloc[-1].item()
    except:
        volume_spike = False
        latest_rsi = 50
    return volume_spike, latest_rsi

def v103_score(E, T=1.2, S=1.1, D=1.0, M=1.0, G=1.0, P=1.0, F=1.1, V=1.0, R=1.0, B=1.0, reversal=1.0):
    E_enhanced = E + 1 if E > 0 else E - 1 if E < 0 else 0
    score = E_enhanced * T * S * D * M * G * P * F * V * R * B * reversal
    return round(score, 2)

def interpret(score):
    if score > 1.5:
        return "UP"
    elif score < -1.5:
        return "DOWN"
    else:
        return "SIDEWAYS"

def validate_ticker(ticker):
    try:
        test = yf.Ticker(ticker).info
        return 'shortName' in test
    except:
        return False

# --- STREAMLIT UI ---
st.set_page_config(page_title="V10.3 Stock Predictor", page_icon="📈")
st.title("📈 V10.3 Stock Prediction AI")
st.caption("Predict tomorrow's stock movement using real news and market signals.")

user_input = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA, SHOP.TO)").upper()

if user_input:
    with st.spinner("🔄 Fetching data and calculating prediction..."):
        if not validate_ticker(user_input):
            st.error("Invalid or unsupported ticker. Please try another.")
        else:
            st.subheader("🔍 Recent News")
            headlines = get_news_headlines(user_input)
            if headlines:
                for h in headlines:
                    st.markdown(f"- {h}")
            else:
                st.markdown("- No news found.")

            st.subheader("🧠 Sentiment Analysis")
            E = get_sentiment_score(headlines)
            st.write(f"Emotion Score (E): {E}")

            st.subheader("📊 Market Data")
            volume_spike, latest_rsi = get_stock_data(user_input)
            V = 1.2 if volume_spike else 1.0
            R = 0.9 if latest_rsi > 70 or latest_rsi < 30 else 1.0
            st.write(f"Volume Spike: {volume_spike}")
            st.write(f"RSI: {latest_rsi:.2f}")

            st.subheader("📈 V10.3 Prediction")
            score = v103_score(E, V=V, R=R)
            prediction = interpret(score)
            confidence = "★★★☆☆" if abs(score) < 2 else "★★★★☆" if abs(score) < 4 else "★★★★★"

            st.metric(label="Prediction", value=prediction)
            st.metric(label="Score", value=score)
            st.metric(label="Confidence", value=confidence)
