from helper import create_wordcloud, sentiment_analysis, topic_modeling
import streamlit as st
import preprocessor,helper
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import logging
import datetime
from matplotlib import font_manager as fm
from io import BytesIO
from helper import devanagari_font_prop, emoji_font_prop



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(layout="wide")
# Add custom CSS for larger font and green color
st.markdown(
    """
    <style>
    .big-font {
        font-size:30px !important;
        color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

sns.set_theme(style="whitegrid", palette="pastel")
st.sidebar.title("📊 Whatsapp Chat Analyzer")
st.sidebar.subheader("Upload your exported WhatsApp chat (.txt)")
st.sidebar.info("Analyze chat stats such as message frequency, most active users, and more.")

uploaded_file = st.sidebar.file_uploader("Upload WhatsApp chat (.txt)", type=["txt"])
if uploaded_file is not None:
    
        st.text("File Preview:")
        st.code(uploaded_file.getvalue().decode("utf-8")[:500], language="text")
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf-8")
        df = preprocessor.preprocess(data)
        
        # Date filter inputs
        import datetime
        st.sidebar.subheader("Filter by Date Range:")
        start_date = st.sidebar.date_input("Start Date", datetime.date(2024, 1, 1))
        end_date = st.sidebar.date_input("End Date", datetime.date.today())

        # Filter the DataFrame based on selected dates
        df['datetime'] = pd.to_datetime(df['datetime'])  # Ensure 'datetime' column is datetime format
        filtered_df = df[(df['datetime'] >= pd.Timestamp(start_date)) & (df['datetime'] <= pd.Timestamp(end_date))]

        # Replace df with filtered_df for all analyses
        df = filtered_df  # Use filtered DataFrame for analysis

        # fetch unique users
        user_list = df['user'].unique().tolist()
        # Check if 'group_notification' exists in the list before removing it
        if 'group_notification' in user_list:
            user_list.remove('group_notification')

        user_list.sort()
        user_list.insert(0,"Overall")

        selected_user = st.sidebar.selectbox("Show analysis wrt",user_list)

        if st.sidebar.button("Show Analysis"):
            tab1, tab2, tab3 = st.tabs(["Overview", "Timelines", "Analysis"])

            # Stats Area
            num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user,df)
            with tab1:
                st.title("Top Statistics")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.header("Top Messages")
                    st.title(num_messages)
                with col2:
                    st.header("Total Words")
                    st.title(words)
                with col3:
                    st.header("Media Shared")
                    st.title(num_media_messages)
                with col4:
                    st.header("Links Shared")
                    st.title(num_links)
                
            
                # Progress bar for media contribution
                if num_messages > 0:  # Avoid division by zero
                    progress = int((num_media_messages / num_messages) * 100)
                    st.progress(progress)
                    st.text(f"Media contributes {progress}% of all messages.")
                else:
                    st.warning("No messages to analyze for media contribution.")

            # monthly timeline
            with tab2:
                st.title("🗓️Message Timelines")
                timeline = helper.monthly_timeline(selected_user,df)
                fig = px.line(timeline, x='time', y='message', title='Monthly Timeline', markers=True)
                st.plotly_chart(fig)

                # activity map
                st.title('Activity Map')
                col1,col2 = st.columns(2)

                with col1:
                    st.header("Most busy day")
                    busy_day = helper.week_activity_map(selected_user,df)
                    st.bar_chart(data=busy_day)
                    
                with col2:
                    st.header("Most busy month")
                    busy_month = helper.month_activity_map(selected_user, df)
                    st.bar_chart(data=busy_month)
                    
                st.title("📊 Activity Heatmap")
                user_heatmap = helper.activity_heatmap(selected_user,df)
                if user_heatmap.empty or np.isnan(user_heatmap).all().all():
                    st.error("The heatmap data is empty or contains invalid values.")
                else:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.heatmap(user_heatmap, cmap='YlGnBu', linewidths=0.5, annot=True, fmt='.0f', ax=ax)
                    ax.set_title("Activity Heatmap", fontsize=16)
                    ax.set_xlabel("Hour of the Day", fontsize=12)
                    ax.set_ylabel("Day of the Week", fontsize=12)
                    st.pyplot(fig)

                # finding the busiest users in the group(Group level)
                if selected_user == 'Overall':
                    st.title('Most Busy Users')
                    x,new_df = helper.most_busy_users(df)
                    fig, ax = plt.subplots()

                    col1, col2 = st.columns(2)

                    with col1:
                        ax.bar(x.index, x.values,color='red')
                        plt.xticks(rotation='vertical')
                        st.pyplot(fig)
                    with col2:
                        st.dataframe(new_df)

            # WordCloud
            with tab3:
                st.title("Emoji & Word Analysis")
                st.title("Wordcloud")
                
                # Generate the WordCloud
                df_wc = create_wordcloud(selected_user, df)
        
                if isinstance(df_wc, str):
                    st.error(df_wc)
                else:
                    # Display the WordCloud using Matplotlib
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(df_wc, interpolation='bilinear')
                    ax.axis("off")  # Hide axes for better appearance
                    st.pyplot(fig)
                    
                    # Save WordCloud to a file-like object
                    from io import BytesIO
                    buffer = BytesIO()
                    fig.savefig(buffer, format="png", bbox_inches="tight")
                    buffer.seek(0)
                    
                    # Add a download button
                    st.download_button(
                        label="Download WordCloud",
                        data=buffer,
                        file_name="wordcloud.png",
                        mime="image/png"
                    )

                # most common words
                prop = emoji_font_prop


                # Fetch most common words
                most_common_df = helper.most_common_words(selected_user, df)
                st.write(most_common_df.head())

                # Create a horizontal bar plot
                fig, ax = plt.subplots()

                # Plot data and apply the emoji font
                ax.barh(most_common_df[0], most_common_df[1], color='skyblue')
                ax.set_title('🔥Most Common Words', fontproperties=devanagari_font_prop)
                ax.set_xlabel("Count",  fontproperties=devanagari_font_prop)
                ax.set_ylabel("Words",  fontproperties=devanagari_font_prop)
                ax.set_yticks(range(len(most_common_df[0])))
                ax.set_yticklabels(most_common_df[0],  fontproperties=devanagari_font_prop)  # Apply emoji font to y-axis labels
                plt.xticks(rotation='vertical', fontproperties=devanagari_font_prop)
                st.pyplot(fig)

                # Emoji Analysis
                emoji_df = helper.emoji_helper(selected_user, df)
                st.title("Emoji Analysis")

                col1, col2 = st.columns(2)

                # Display Emoji Data in a Table
                with col1:
                    st.dataframe(emoji_df)

                #Pie Chart
                with col2:
                    emoji_font = emoji_font_prop
                    fig, ax = plt.subplots(figsize=(8, 8))  # Adjust figure size for better readability
                    
                    # Extract top 5 emojis and their usage counts
                    emoji_labels = emoji_df[0].head()  # First column (emojis)
                    emoji_values = emoji_df[1].head()  # Second column (frequencies)
                    
                    # Plot the pie chart
                    wedges, texts, autotexts = ax.pie(
                        emoji_values,
                        labels=emoji_labels,
                        autopct="%1.1f%%",  # Display percentages
                        startangle=90,  # Align largest slice at top
                        pctdistance=0.95,  # Position percentages closer to the center
                        colors=plt.cm.Set3.colors,  # Use a visually distinct color palette
                    )

                    # Style percentage labels
                    for autotext in autotexts:
                        autotext.set_fontsize(6)
                        autotext.set_color("black")
                        autotext.set_fontweight("bold")
                        autotext.set_fontproperties(devanagari_font_prop)

                    # Style emoji labels using custom emoji-compatible font
                    for text in texts:
                        text.set_fontproperties(emoji_font_prop)
                        text.set_fontsize(14)
                        text.set_color("black")

                    # Add a central circle to make it a donut chart (optional)
                    # center_circle = plt.Circle((0, 0), 0.70, fc="white")
                    # fig.gca().add_artist(center_circle)

                    # Add a title to the pie chart
                    ax.set_title("Top Emoji Usage", fontsize=16, weight="bold",  fontproperties=devanagari_font_prop)

                    # Display the chart in Streamlit
                    st.pyplot(fig)
                
                # Sentiment Analysis
                st.title("📈 Sentiment Analysis")
                daily_sentiment = sentiment_analysis(df)

                # Plot daily sentiment trends
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(daily_sentiment['only_date'], daily_sentiment['sentiment'], marker='o', linestyle='-', color='purple')
                ax.set_title("Daily Sentiment Trends", fontsize=16)
                ax.set_xlabel("Date", fontsize=12)
                ax.set_ylabel("Sentiment Polarity", fontsize=12)
                st.pyplot(fig)

                # Topic Modeling
                st.title("📚 Topic Modeling")
                num_topics = st.slider("Number of Topics", min_value=2, max_value=10, value=5, step=1)
                topics = topic_modeling(df, num_topics=num_topics)
                
                # Display topics
                for topic, words in topics.items():
                    st.write(f"**{topic}:** {', '.join(words)}")






