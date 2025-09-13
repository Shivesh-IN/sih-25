# app.py

import streamlit as st
import pandas as pd
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64, json

st.set_page_config(page_title="Comment Analyzer", layout="wide")
st.title("Comment Analysis: Sentiment, Summary, and Word Cloud")

# 1. Initialize Hugging Face pipelines (with caching to avoid reloading on each run).
@st.cache_resource(allow_output_mutation=True)
def load_pipelines():
    sentiment_pipe = pipeline("sentiment-analysis")
    summary_pipe = pipeline("summarization", model="facebook/bart-large-cnn")
    return sentiment_pipe, summary_pipe

classifier, summarizer = load_pipelines()

# 2. File uploader: accept a CSV with a column (e.g. 'comment').
uploaded_file = st.file_uploader("Upload CSV file with comments", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Check for a 'comment' column (adjust name if needed)
    if "comment" not in df.columns:
        st.error("CSV must have a 'comment' column.")
    else:
        st.write("### Raw Data")
        st.dataframe(df)  # display original comments

        # 3. Run sentiment analysis
        texts = df["comment"].astype(str).tolist()
        with st.spinner("Analyzing sentiment..."):
            results = classifier(texts)  # returns list of dicts
        df["sentiment_label"] = [res["label"] for res in results]
        df["sentiment_score"] = [res["score"] for res in results]

        # 4. Run summarization
        with st.spinner("Generating summaries..."):
            sum_results = summarizer(texts, max_length=50, min_length=20, do_sample=False)
        df["summary"] = [res["summary_text"] for res in sum_results]

        # 5. Display analyzed data
        st.write("### Analysis Results")
        st.dataframe(df)

        # 6. Sentiment distribution bar chart
        st.write("### Sentiment Distribution")
        sentiment_counts = df["sentiment_label"].value_counts()
        fig, ax = plt.subplots()
        ax.bar(sentiment_counts.index, sentiment_counts.values, color=['#8b0000','#228b22'])
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        ax.set_title("Positive vs Negative Comments")
        st.pyplot(fig)

        # 7. Word Cloud from all comments
        st.write("### Word Cloud of Comments")
        all_text = " ".join(texts)
        if all_text.strip() != "":
            wc = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.imshow(wc, interpolation='bilinear')
            ax2.axis("off")
            st.pyplot(fig2)
        else:
            st.write("No text for word cloud.")

        # 8. Download results as CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results as CSV", csv, "analysis_results.csv", "text/csv")

        # 9. (Optional) Save to Firebase
        if st.checkbox("Save results to Firebase Firestore"):
            try:
                # Load credentials from streamlit secrets
                cred_dict = json.loads(st.secrets["firebase"])
            except:
                st.error("Firebase credentials not found in secrets. Skipping save.")
            else:
                from firebase_admin import credentials, firestore, initialize_app
                # Initialize app (only once)
                try:
                    initialize_app(credentials.Certificate(cred_dict))
                except ValueError:
                    pass
                db = firestore.client()
                # Save as a new document with a timestamp
                db.collection("comment_analysis").add({"results": df.to_dict(orient="records")})
                st.success("Results saved to Firebase!")

