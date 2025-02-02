from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import matplotlib.pyplot as plt

extract = URLExtract()

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'].str.contains('<Media omitted>', na=False)].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)

def most_busy_users(df):
    user_counts = df['user'].value_counts().head()
    user_percentages = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index()
    user_percentages.columns = ['name', 'percent']
    return user_counts, user_percentages

def create_wordcloud(selected_user, df):
    # Open stop words file and read its contents
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = f.read().splitlines()  # Ensure stop words are split into a list

    # Filter data based on selected user
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['user'] != 'group_notification']
    df = df[df['message'] != '<Media omitted>']    

    def remove_stop_words(message):
        words = message.lower().split()
        return " ".join(word for word in words if word not in stop_words)    
    
    df['message'] = df['message'].apply(remove_stop_words)
    text = df['message'].str.cat(sep=" ")

    if not text.strip():
        return "No valid words available to generate a word cloud."
    
    # Create WordCloud object
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    return wc.generate(text)

def most_common_words(selected_user,df):
    with open('stop_hinglish.txt','r') as f:
        stop_words = f.read().splitlines()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['user'] != 'group_notification']
    df = df[df['message'] != '<Media omitted>']

    words = []

    for message in df['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    return pd.DataFrame(Counter(words).most_common(20))            

def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    return pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df.groupby('only_date').count()['message'].reset_index()

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    """
    Create a heatmap of activity based on days and hours.

    Args:
        selected_user (str): Selected user or "Overall".
        df (DataFrame): Preprocessed chat data.

    Returns:
        DataFrame: Pivot table for heatmap visualization.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df['hour'] = df['datetime'].dt.hour
    heatmap_data = df.pivot_table(index='day_name', columns='hour', values='message', aggfunc='count').fillna(0)
    return heatmap_data


from textblob import TextBlob

def sentiment_analysis(df):
    # Calculate sentiment polarity for each message
    df['sentiment'] = df['message'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Aggregate sentiment by day
    daily_sentiment = df.groupby('only_date')['sentiment'].mean().reset_index()
    
    return daily_sentiment


def topic_modeling(df, num_topics):
    # Preprocessing: Tokenize and remove stop words
    df['tokens'] = df['message'].apply(lambda x: [word.lower() for word in x.split() if word.isalpha()])
    
    # Create a Gensim dictionary and corpus
    dictionary = Dictionary(df['tokens'])
    corpus = [dictionary.doc2bow(text) for text in df['tokens']]
    
    # Train the LDA model
    lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    
    # Get topics
    topics = lda_model.show_topics(num_topics=num_topics, num_words=5, formatted=False)
    
    # Prepare topics for visualization
    topic_words = {f"Topic {i}": [word for word, _ in topic[1]] for i, topic in enumerate(topics)}

    return topic_words
