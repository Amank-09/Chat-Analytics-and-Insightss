import re
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess(data):
    """Preprocess the WhatsApp chat data and return a structured DataFrame."""
    # Define a regex pattern for WhatsApp messages
    pattern = r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}\s[APap][Mm]) - ([^:]+): (.*)$'
    messages = []
    for line in data.splitlines():
        match = re.match(pattern, line)
        if match:
            date, time, user, message = match.groups()
            messages.append([date, time, user, message])
        else:
            if messages:
                messages[-1][-1] += " " + line.strip()

    df = pd.DataFrame(messages, columns=['date', 'time', 'user', 'message'])

    # Convert date and time into datetime format
    try:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d/%m/%y %I:%M %p')
    except Exception as e:
        logging.error(f"Date parsing error: {e}")

    # Drop unnecessary columns
    df.drop(columns=['date', 'time'], inplace=True)

    # Add additional features
    df['day_name'] = df['datetime'].dt.day_name()
    df['only_date'] = df['datetime'].dt.date
    df['month'] = df['datetime'].dt.month_name()
    df['year'] = df['datetime'].dt.year
    df['month_num'] = df['datetime'].dt.month
    df['period'] = df['datetime'].dt.strftime('%H:%M')

    return df

# Usage example
if __name__ == "__main__":
    try:
        with open('WhatsApp Chat with Placement 2025 EEE.txt', 'r', encoding='utf-8') as file:
            raw_data = file.read()

        df = preprocess(raw_data)
        logging.info("Preprocessing completed successfully!")
        df.to_csv('processed_chat.csv', index=False)
        logging.info("Processed chat saved to processed_chat.csv.")
    except FileNotFoundError:
        logging.error("Chat file not found.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")