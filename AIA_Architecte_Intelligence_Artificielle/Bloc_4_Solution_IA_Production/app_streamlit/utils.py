import pandas as pd
from sqlalchemy import create_engine
import os

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

def load_scoring():
    query = """
        SELECT id, created_at, joy, anger, sadness, surprise, disgust, fear, neutral, main_emotion
        FROM reddit_scoring
        ORDER BY created_at ASC;
    """
    return pd.read_sql(query, engine)
