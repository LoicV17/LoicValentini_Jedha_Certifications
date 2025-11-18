import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataQualityPreset
from sqlalchemy import create_engine

DATABASE_URL = os.getenv("DATABASE_URL")

def main():
    engine = create_engine(DATABASE_URL)

    df = pd.read_sql("SELECT * FROM reddit_scoring ORDER BY created_at DESC LIMIT 500;", engine)

    report = Report(metrics=[DataQualityPreset()])
    report.run(df)

    os.makedirs("reports/jenkins", exist_ok=True)
    report.save_html("reports/jenkins/evidently_report.html")

if __name__ == "__main__":
    main()
