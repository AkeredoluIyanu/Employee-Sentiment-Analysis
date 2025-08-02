# Employee Sentiment Analysis

# Project Overview
This project analyzes employee messages to classify sentiment, track engagement trends, identify flight risks, and build a predictive model to forecast sentiment scores.

# Folder Structure
- `data`: Raw and processed datasets.
- `notebooks`: Jupyter notebooks for each stage of the project.
- `visualizations`: Charts and graphs summarizing analysis and model performance.
- `final_report.docx`: Detailed project report.
- `requirements.txt`: Python dependencies.

## Key Results
- Sentiment labeled for each message using TextBlob polarity.
- Monthly sentiment scores computed per employee.
- Identified top 3 positive and negative employees monthly.
- Flagged flight risk employees sending 4+ negative messages in 30 days.
- Linear regression model predicting sentiment scores with RMSE ~0.42 and R² ~0.42.

## How to Run
1. Clone the repository.
2. Install dependencies:
