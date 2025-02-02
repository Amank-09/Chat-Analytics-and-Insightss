from collections import Counter
            prop = fm.FontProperties(fname=emoji_font_path)

            # Fetch most common words
            words = df['message'].str.cat(sep=' ').split()  # Combine all messages and split into words
            most_common_words = Counter(words).most_common(10)  # Get the 10 most common words
            # Convert to DataFrame with explicit column names
            most_common_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
            print(most_common_df.head())