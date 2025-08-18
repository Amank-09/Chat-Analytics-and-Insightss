import os
from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import warnings
from font_config import emoji_font_prop


extract = URLExtract()

try:
    import nltk
    nltk.data.find("tokenizers/punkt")
except LookupError:
    import nltk
    nltk.download("punkt")

BASE_DIR = os.path.dirname(__file__)
STOPWORDS_PATH = os.path.join(BASE_DIR, "stop_hinglish.txt")


def fetch_stats(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    num_messages = df.shape[0]

    words = []
    for message in df["message"].dropna().astype(str):
        words.extend(message.split())

    num_media_messages = df[df["message"].str.contains("<Media omitted>", na=False)].shape[0]

    links = []
    for message in df["message"].dropna().astype(str):
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)


def most_busy_users(df):
    user_counts = df["user"].value_counts().head()
    user_percentages = round((df["user"].value_counts() / df.shape[0]) * 100, 2).reset_index()
    user_percentages.columns = ["name", "percent"]
    return user_counts, user_percentages


def create_wordcloud(selected_user, df):
    if os.path.exists(STOPWORDS_PATH):
        with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
            stop_words = f.read().splitlines()
    else:
        stop_words = []

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    df = df[df["user"] != "group_notification"]
    df = df[df["message"] != "<Media omitted>"]

    def remove_stop_words(message):
        if not isinstance(message, str):
            return ""
        words = message.lower().split()
        return " ".join(word for word in words if word not in stop_words)

    df = df.copy()
    df["message"] = df["message"].apply(remove_stop_words)
    text = df["message"].str.cat(sep=" ")

    if not text.strip():
        return "No valid words available to generate a word cloud."

    wc = WordCloud(
        width=500,
        height=500,
        min_font_size=10,
        background_color="white",
        font_path=emoji_font_prop.get_file() if emoji_font_prop.get_file() else None
    )

    return wc.generate(text)


def most_common_words(selected_user, df):
    if os.path.exists(STOPWORDS_PATH):
        with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
            stop_words = f.read().splitlines()
    else:
        stop_words = []

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    df = df[df["user"] != "group_notification"]
    df = df[df["message"] != "<Media omitted>"]

    words = []
    for message in df["message"].dropna().astype(str):
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    return pd.DataFrame(Counter(words).most_common(20))


def emoji_helper(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    emojis_list = []
    for message in df["message"].dropna().astype(str):
        try:
            extracted = [e["emoji"] for e in emoji.emoji_list(message)]
        except Exception:
            extracted = [c for c in message if c in emoji.EMOJI_DATA] if hasattr(emoji, "EMOJI_DATA") else []
        emojis_list.extend(extracted)

    return pd.DataFrame(Counter(emojis_list).most_common(len(Counter(emojis_list))))


def monthly_timeline(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    timeline = df.groupby(["year", "month_num", "month"]).count()["message"].reset_index()
    timeline["time"] = timeline["month"] + "-" + timeline["year"].astype(str)

    return timeline


def daily_timeline(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    return df.groupby("only_date").count()["message"].reset_index()


def week_activity_map(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    return df["day_name"].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    return df["month"].value_counts()


def activity_heatmap(selected_user, df):
    """
    Create a heatmap pivot table of activity based on days and hours.
    """
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    if "datetime" in df.columns:
        df = df.copy()
        df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
        heatmap_data = df.pivot_table(index="day_name", columns="hour", values="message", aggfunc="count").fillna(0)
        return heatmap_data
    else:
        return pd.DataFrame()


from textblob import TextBlob


def sentiment_analysis(df):
    df = df.copy()
    df["message"] = df["message"].fillna("").astype(str)

    def _safe_polarity(text):
        try:
            return TextBlob(text).sentiment.polarity
        except Exception:
            return 0.0

    df["sentiment"] = df["message"].apply(_safe_polarity)

    if "only_date" in df.columns:
        daily_sentiment = df.groupby("only_date")["sentiment"].mean().reset_index()
    else:
        daily_sentiment = pd.DataFrame({"sentiment": [df["sentiment"].mean()]})

    return daily_sentiment


def topic_modeling(df, num_topics):
    df = df.copy()
    df["message"] = df["message"].fillna("").astype(str)

    df["tokens"] = df["message"].apply(lambda x: [word.lower() for word in x.split() if word.isalpha()])

    dictionary = Dictionary(df["tokens"])
    corpus = [dictionary.doc2bow(text) for text in df["tokens"]]

    lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    topics = lda_model.show_topics(num_topics=num_topics, num_words=5, formatted=False)

    topic_words = {f"Topic {i}": [word for word, _ in topic[1]] for i, topic in enumerate(topics)}

    return topic_words









