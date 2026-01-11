import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎥",
    layout="wide"
)

st.title("Movie Recommender System")
st.write("Pick multiple movies you like and get personalized recommendations.")


@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    df = df[df["year"] >= 1950]
    df = df[df["votes"] >= 1000]

    df = df.reset_index(drop=True)
    return df

df = load_data()


def year_bucket(year):
    return f"{(year // 10) * 10}s"

df["profile"] = (
    df["genres"].astype(str) + " " +
    df["genres"].astype(str) + " " +   
    df["year"].apply(year_bucket)
)


@st.cache_resource
def build_vectors(profiles):
    vectorizer = TfidfVectorizer(lowercase=True)
    X = vectorizer.fit_transform(profiles)
    return vectorizer, X

vectorizer, X = build_vectors(df["profile"])


st.sidebar.header("Settings")

SIM_WEIGHT = st.sidebar.slider(
    "Similarity weight",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05
)

RATING_WEIGHT = 1.0 - SIM_WEIGHT

TOP_K = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)


liked_movies = st.multiselect(
    "Select movies you like:",
    options=df["title"].tolist()
)


if st.button("🍿 Recommend"):

    if len(liked_movies) == 0:
        st.warning("Please select at least one movie.")
    else:
        liked_idxs = df[df["title"].isin(liked_movies)].index

        user_vector = X[liked_idxs].mean(axis=0)
        user_vector = np.asarray(user_vector).ravel()

        df["similarity"] = cosine_similarity(
            user_vector.reshape(1, -1),
            X
        ).flatten()

        df.loc[liked_idxs, "similarity"] = -1

        df["rating_norm"] = df["rating"] / 10

        df["final_score"] = (
            df["similarity"] * SIM_WEIGHT +
            df["rating_norm"] * RATING_WEIGHT
        )

        recommendations = (
            df.sort_values("final_score", ascending=False)
              .head(TOP_K)
        )

  
        st.subheader(" Recommended Movies")

        st.dataframe(
            recommendations[
                ["title", "genres", "year", "rating", "similarity", "final_score"]
            ],
            use_container_width=True
        )

        st.caption(
            "Scores combine similarity to your taste and IMDb rating quality."
        )
