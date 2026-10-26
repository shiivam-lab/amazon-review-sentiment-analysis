import os

import pandas as pd
import plotly.express as px
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Amazon Review Sentiment Analysis", layout="wide")

DATA_FILE = "reviews.csv"


@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    elif os.path.exists(DATA_FILE):
        data = pd.read_csv(DATA_FILE)
    else:
        return pd.DataFrame()

    if "Review" not in data.columns:
        st.error("The dataset must contain a 'Review' column.")
        return pd.DataFrame()

    data["Review"] = data["Review"].fillna("").astype(str)

    analyzer = SentimentIntensityAnalyzer()
    if "Sentiment Score" not in data.columns:
        data["Sentiment Score"] = data["Review"].apply(
            lambda review: analyzer.polarity_scores(review)["compound"]
        )

    if "Sentiment" not in data.columns:
        def label_sentiment(score):
            if score > 0.01:
                return "Positive"
            if score < -0.01:
                return "Negative"
            return "Neutral"

        data["Sentiment"] = data["Sentiment Score"].apply(label_sentiment)

    return data


st.sidebar.title("Amazon Review Analytics")
uploaded_file = st.sidebar.file_uploader("Upload a review CSV (optional)", type=["csv"])
df = load_data(uploaded_file)

if df.empty:
    st.title("Amazon Customer Review Sentiment Analysis")
    st.info("Add reviews.csv to the project folder or upload a CSV using the sidebar.")
    st.stop()

menu = st.sidebar.selectbox(
    "Navigate",
    ["Home", "Review Analysis", "Visualizations", "ML Prediction", "Business Dashboard"],
)

# Optional interactive filters
filtered = df.copy()
if menu in ["Visualizations", "ML Prediction", "Business Dashboard"]:
    st.sidebar.header("Filters")
    if "Product Name" in filtered.columns:
        products = ["All"] + sorted(filtered["Product Name"].dropna().astype(str).unique().tolist())
        product = st.sidebar.selectbox("Product", products)
        if product != "All":
            filtered = filtered[filtered["Product Name"].astype(str) == product]

    if "Gender" in filtered.columns:
        genders = ["All"] + sorted(filtered["Gender"].dropna().astype(str).unique().tolist())
        gender = st.sidebar.selectbox("Gender", genders)
        if gender != "All":
            filtered = filtered[filtered["Gender"].astype(str) == gender]

    sentiment = st.sidebar.selectbox("Sentiment", ["All", "Positive", "Negative", "Neutral"])
    if sentiment != "All":
        filtered = filtered[filtered["Sentiment"] == sentiment]

if menu == "Home":
    st.title("Amazon Customer Review Sentiment Analysis")
    st.write(
        "Explore customer feedback, review sentiment, product-level patterns, "
        "and potential opportunities to improve customer experience."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Reviews", f"{len(df):,}")
    counts = df["Sentiment"].value_counts()
    c2.metric("Most Common Sentiment", counts.idxmax() if not counts.empty else "N/A")
    c3.metric("Products Covered", df["Product Name"].nunique() if "Product Name" in df.columns else "N/A")
    st.info("Use the sidebar to explore reviews, visualizations, model predictions, and business insights.")

elif menu == "Review Analysis":
    st.title("Review Analysis")
    st.write("Summary of the available review dataset.")
    counts = df["Sentiment"].value_counts().reindex(["Positive", "Negative", "Neutral"], fill_value=0)
    summary = pd.DataFrame({
        "Sentiment": counts.index,
        "Reviews": counts.values,
        "Percentage": (counts.values / len(df) * 100).round(2) if len(df) else 0,
    })
    st.dataframe(summary, use_container_width=True)
    st.subheader("Search Reviews")
    keyword = st.text_input("Search by a word or phrase")
    search_df = filtered.copy()
    if keyword:
        search_df = search_df[search_df["Review"].str.contains(keyword, case=False, na=False)]
    st.caption(f"{len(search_df):,} review(s) found")
    st.dataframe(search_df, use_container_width=True)

elif menu == "Visualizations":
    st.title("Review Visualizations")
    if filtered.empty:
        st.warning("No reviews match the selected filters.")
        st.stop()

    c1, c2 = st.columns(2)
    sentiment_counts = filtered["Sentiment"].value_counts().reindex(
        ["Positive", "Negative", "Neutral"], fill_value=0
    )
    with c1:
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            hole=0.4,
            title="Sentiment Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(
            filtered,
            x="Sentiment Score",
            color="Sentiment",
            title="Sentiment Score Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    if "Product Name" in filtered.columns:
        product_sentiment = (
            filtered.groupby(["Product Name", "Sentiment"])
            .size().reset_index(name="Reviews")
        )
        fig = px.bar(
            product_sentiment, x="Product Name", y="Reviews", color="Sentiment",
            barmode="group", title="Product-wise Sentiment Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

    negative = filtered[filtered["Sentiment"] == "Negative"]
    positive = filtered[filtered["Sentiment"] == "Positive"]
    keywords = ["quality", "sound", "battery", "charging", "camera",
                "performance", "display", "heating", "hanging", "slow"]
    problem_rows = [
        {"Keyword": word, "Reviews": int(negative["Review"].str.contains(word, case=False, na=False).sum())}
        for word in keywords
    ]
    feature_rows = [
        {"Keyword": word, "Reviews": int(positive["Review"].str.contains(word, case=False, na=False).sum())}
        for word in keywords
    ]
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Common Terms in Negative Reviews")
        st.dataframe(pd.DataFrame(problem_rows).sort_values("Reviews", ascending=False), use_container_width=True)
    with c2:
        st.subheader("Common Terms in Positive Reviews")
        st.dataframe(pd.DataFrame(feature_rows).sort_values("Reviews", ascending=False), use_container_width=True)

    if "Age" in filtered.columns:
        age_df = filtered.copy()
        age_df["Age"] = pd.to_numeric(age_df["Age"], errors="coerce")
        age_df = age_df.dropna(subset=["Age"])
        if not age_df.empty:
            age_df["Age Group"] = pd.cut(
                age_df["Age"], bins=[0, 20, 30, 40, 50, 120],
                labels=["20 or below", "21-30", "31-40", "41-50", "51+"],
                include_lowest=True,
            )
            age_data = age_df.groupby(["Age Group", "Sentiment"], observed=False).size().reset_index(name="Reviews")
            fig = px.bar(age_data, x="Age Group", y="Reviews", color="Sentiment",
                         barmode="group", title="Sentiment by Age Group")
            st.plotly_chart(fig, use_container_width=True)

elif menu == "ML Prediction":
    st.title("Machine Learning Sentiment Prediction")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.model_selection import train_test_split

    ml_df = filtered.dropna(subset=["Review", "Sentiment"]).copy()
    ml_df["Review"] = ml_df["Review"].astype(str)
    if len(ml_df) < 10 or ml_df["Sentiment"].nunique() < 2:
        st.warning("At least 10 reviews and two sentiment classes are required for model evaluation.")
    else:
        class_counts = ml_df["Sentiment"].value_counts()
        stratify = ml_df["Sentiment"] if class_counts.min() >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(
            ml_df["Review"], ml_df["Sentiment"], test_size=0.2,
            random_state=42, stratify=stratify
        )
        vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
        x_train_vec = vectorizer.fit_transform(x_train)
        x_test_vec = vectorizer.transform(x_test)
        model = LogisticRegression(max_iter=1000)
        model.fit(x_train_vec, y_train)
        predictions = model.predict(x_test_vec)
        accuracy = accuracy_score(y_test, predictions)
        c1, c2, c3 = st.columns(3)
        c1.metric("Test Accuracy", f"{accuracy * 100:.2f}%")
        c2.metric("Training Reviews", f"{len(y_train):,}")
        c3.metric("Testing Reviews", f"{len(y_test):,}")
        st.caption("Accuracy is calculated on this run's held-out test split; it is not a guarantee of real-world performance.")
        labels = sorted(ml_df["Sentiment"].unique())
        matrix = confusion_matrix(y_test, predictions, labels=labels)
        fig = px.imshow(matrix, text_auto=True, x=labels, y=labels,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Classification report"):
            report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        st.subheader("Try a New Review")
        review = st.text_area("Enter a customer review")
        if st.button("Predict Sentiment"):
            if not review.strip():
                st.warning("Enter a review first.")
            else:
                prediction = model.predict(vectorizer.transform([review]))[0]
                st.success(f"Predicted sentiment: {prediction}")

elif menu == "Business Dashboard":
    st.title("Amazon Review Business Dashboard")
    total = len(filtered)
    counts = filtered["Sentiment"].value_counts()
    positive = int(counts.get("Positive", 0))
    negative = int(counts.get("Negative", 0))
    neutral = int(counts.get("Neutral", 0))
    avg_score = filtered["Sentiment Score"].mean() if total else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Reviews", f"{total:,}")
    c2.metric("Positive", f"{positive / total * 100:.2f}%" if total else "0.00%")
    c3.metric("Negative", f"{negative / total * 100:.2f}%" if total else "0.00%")
    c4.metric("Neutral", f"{neutral / total * 100:.2f}%" if total else "0.00%")
    c5.metric("Average Sentiment Score", f"{avg_score:.2f}" if total else "0.00")

    if total:
        sentiment_summary = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Reviews": [positive, negative, neutral],
        })
        fig = px.pie(sentiment_summary, values="Reviews", names="Sentiment",
                     hole=0.45, title="Overall Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)

        if "Product Name" in filtered.columns:
            product_data = filtered.groupby(["Product Name", "Sentiment"]).size().reset_index(name="Reviews")
            fig = px.bar(product_data, x="Product Name", y="Reviews", color="Sentiment",
                         barmode="group", title="Product Performance by Sentiment")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Business Recommendations")
        if negative / total > 0.30:
            st.warning("Negative sentiment is high in this selection. Investigate recurring complaints and prioritize product improvements.")
        elif negative / total > 0.20:
            st.info("Review negative feedback and look for recurring product or service issues.")
        else:
            st.success("Positive sentiment is the majority in this selection. Continue monitoring negative feedback and customer needs.")
    else:
        st.warning("No reviews match the selected filters.")
