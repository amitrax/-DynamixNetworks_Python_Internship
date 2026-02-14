import streamlit as st
import pandas as pd
from datetime import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="AI Sentiment Chatbot", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "logs" not in st.session_state:
    st.session_state.logs = []

def analyze_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return "Positive", "😊", "I'm glad to hear that! How can I help you more?"
    elif score <= -0.05:
        return "Negative", "😟", "I'm sorry you're feeling this way. I'm here to help."
    else:
        return "Neutral", "😐", "I see. What else can I do for you?"

st.title("🤖 AI Chatbot with Sentiment Analysis")
st.markdown("---")

chat_col, dash_col = st.columns([2, 1])

with chat_col:
    st.subheader("Chat Interface")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Type your message..."):
        sentiment, emoji, bot_prefix = analyze_sentiment(prompt)
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        bot_response = f"{bot_prefix} (Detected Tone: {sentiment} {emoji})"
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.markdown(bot_response)

        new_log = {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "User Message": prompt,
            "Sentiment": sentiment,
            "Emoji": emoji
        }
        st.session_state.logs.append(new_log)

with dash_col:
    st.subheader("📊 Analytics Dashboard")
    if st.session_state.logs:
        df = pd.DataFrame(st.session_state.logs)
        st.dataframe(df, use_container_width=True)
        
        st.write("Sentiment Distribution")
        st.bar_chart(df['Sentiment'].value_counts())
    else:
        st.info("Start chatting to see analysis!")

if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.session_state.logs = []
    st.rerun()