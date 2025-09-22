import tweepy
import requests
import time
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_URL = "http://127.0.0.1:8000/predict-retrieval"
LAST_SEEN_ID_FILE = "last_seen_id.txt"

def read_last_seen_id():
    try:
        with open(LAST_SEEN_ID_FILE, "r") as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None

def write_last_seen_id(tweet_id):
    with open(LAST_SEEN_ID_FILE, "w") as f:
        f.write(str(tweet_id))

def run_bot():
    logging.info("Bot is starting...")
    try:
        client = tweepy.Client(
            consumer_key=os.getenv("API_KEY"),
            consumer_secret=os.getenv("API_KEY_SECRET"),
            access_token=os.getenv("ACCESS_TOKEN"),
            access_token_secret=os.getenv("ACCESS_TOKEN_SECRET")
        )
        bot_id = client.get_me().data.id
        logging.info(f"Authenticated as bot with ID: {bot_id}")
    except Exception as e:
        logging.critical(f"Error authenticating with Twitter: {e}")
        return

    since_id = read_last_seen_id()
    while True:
        try:
            mentions = client.get_users_mentions(id=bot_id, since_id=since_id, tweet_fields=["author_id"])
            if mentions.data:
                new_since_id = mentions.meta['newest_id']
                for tweet in mentions.data:
                    if tweet.author_id == bot_id:
                        continue
                    logging.info(f"Found mention ID {tweet.id}: {tweet.text}")
                    response = requests.post(API_URL, json={"text": tweet.text})
                    response.raise_for_status()
                    reply_text = response.json()['reply']
                    client.create_tweet(in_reply_to_tweet_id=tweet.id, text=reply_text)
                    logging.info(f"Replied with: '{reply_text}'")
                write_last_seen_id(new_since_id)
                since_id = new_since_id
            else:
                logging.info("No new mentions found.")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
        logging.info("Waiting for 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    run_bot()
