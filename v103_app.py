# V10.3 Stock Predictor - Terminal Version (Improved Flexibility)
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import sys

# --- CONFIG ---
NEWS_COUNT = 5


def get_news_headlines(ticker):
    try:
        url = f"https://www.google.com/search?q={ticker}+stock+news&tbm=nws"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('div', class_='BNeawe vvjwJb AP7Wnd')
        return [r.get_text() for r in results[:NEWS_COUNT]]
    except Exception as e:
        print(f"[Error fetching news] {e}")
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
        return 2  # Positive
    elif avg_score < -0.25:
        return -2  # Negative
    else:
        return 0  # Neutral


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

    except Exception as e:
        print(f"[Error fetching stock data] {e}")
        volume_spike = False
        latest_rsi = 50  # Neutral fallback

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


def main():
    ticker = input("Enter stock ticker (e.g., AAPL, TSLA, 0700.HK, SHOP.TO): ").upper()

    if not validate_ticker(ticker):
        print("\n[Error] Invalid or unsupported ticker. Please try again.")
        sys.exit()

    print("\n[1] Fetching news...")
    headlines = get_news_headlines(ticker)
    if headlines:
        for h in headlines:
            print(" -", h)
    else:
        print(" - No headlines found.")

    print("\n[2] Analyzing sentiment...")
    E = get_sentiment_score(headlines)
    print("Emotion Score (E):", E)

    print("\n[3] Fetching stock data...")
    volume_spike, latest_rsi = get_stock_data(ticker)
    V = 1.2 if volume_spike else 1.0
    R = 0.9 if latest_rsi > 70 or latest_rsi < 30 else 1.0

    score = v103_score(E, V=V, R=R)
    direction = interpret(score)

    print("\n--- V10.3 Prediction ---")
    print(f"Score: {score}")
    print(f"Prediction: {direction}")
    print("Confidence:", "★★★☆☆" if abs(score) < 2 else "★★★★☆" if abs(score) < 4 else "★★★★★")


if __name__ == "__main__":
    main()

