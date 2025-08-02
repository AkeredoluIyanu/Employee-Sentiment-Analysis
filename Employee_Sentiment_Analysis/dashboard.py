import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import Counter
import re

# Load the labeled sentiment data
df = pd.read_csv("labeled_emails.csv", parse_dates=["Date"])

# Sidebar filters
st.sidebar.title("📬 Email Sentiment Dashboard")
selected_sender = st.sidebar.selectbox("Select Sender", options=["All"] + list(df['Employee'].unique()))
selected_sentiment = st.sidebar.multiselect("Select Sentiment", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"])

# Filter data
filtered_df = df.copy()
if selected_sender != "All":
    filtered_df = filtered_df[filtered_df['Employee'] == selected_sender]
if selected_sentiment:
    filtered_df = filtered_df[filtered_df['Sentiment'].isin(selected_sentiment)]

st.title("📈 Employee Sentiment Analysis Dashboard")
st.markdown(f"Filtered dataset has **{len(filtered_df)} emails**.")

# --- Sentiment Counts
st.subheader("Sentiment Distribution")
sentiment_counts = filtered_df['Sentiment'].value_counts()
st.bar_chart(sentiment_counts)

# --- Word Cloud
st.subheader("Word Cloud")
sentiment_choice = st.selectbox("Choose Sentiment for Word Cloud", ["Positive", "Neutral", "Negative"])
text = " ".join(filtered_df[filtered_df['Sentiment'] == sentiment_choice]['Message'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

# --- Sentiment over time
st.subheader("Sentiment Over Time")
filtered_df['month'] = filtered_df['Date'].dt.to_period("M")
monthly_sentiment = filtered_df.groupby(['month', 'Sentiment']).size().unstack().fillna(0)
st.line_chart(monthly_sentiment)

# --- Top Positive/Negative Senders
st.subheader("Top Positive/Negative Senders")
sender_sentiment = df.groupby(['Employee', 'Sentiment']).size().unstack().fillna(0)
top_positive = sender_sentiment.sort_values('Positive', ascending=False).head(5)
top_negative = sender_sentiment.sort_values('Negative', ascending=False).head(5)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Top Positive Senders**")
    st.dataframe(top_positive[['Positive']])
with col2:
    st.markdown("**Top Negative Senders**")
    st.dataframe(top_negative[['Negative']])

# --- Clustering
st.subheader("Sender Sentiment Clustering")
features = sender_sentiment[['Positive', 'Neutral', 'Negative']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=3, random_state=42)
sender_sentiment['cluster'] = kmeans.fit_predict(features_scaled)

st.dataframe(sender_sentiment.reset_index()[['Employee', 'Positive', 'Neutral', 'Negative', 'cluster']].head(10))



# Download section
st.subheader("📥 Download Labeled Dataset")
csv = df.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name='labeled_sentiment_dataset.csv',
    mime='text/csv'
)