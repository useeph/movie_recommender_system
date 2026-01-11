# IMBD Movie Recommender System

## Live Demo
https://movierecommendersystem-yusif.streamlit.app/

A content-based movie recommendation system that generates personalized suggestions based on user-selected movies using TF-IDF vectorization and cosine similarity. The system combines similarity scoring with IMDb ratings to balance relevance and quality, and is deployed as an interactive web application using Streamlit.

---

## Tech Stack

- Python  
- Streamlit  
- scikit-learn  
- pandas  
- numpy  

---

## Features

- Select multiple movies to define user preferences
- Content-based recommendations using genres and release decade
- Adjustable weighting between similarity score and IMDb rating
- Fast, interactive UI built with Streamlit
- Cached data loading and vectorization for performance

---

## Dataset & Preprocessing

The system is built on IMDb’s public TSV datasets. The raw data is converted into a lightweight `movies.csv` by filtering to movie titles only, merging ratings data, cleaning release years and genres, removing very old titles and low-vote entries, and exporting only the essential fields (title, genres, year, rating, votes). This preprocessing step significantly reduces dataset size and ensures fast loading and efficient recommendation performance.

---

## How It Works

1. **Data Preparation**
   - Movie genres are converted to text features
   - Release years are bucketed into decade groups
   - Genre information is duplicated to emphasize categorical relevance

2. **Vectorization**
   - TF-IDF is applied to the combined genre and decade text profile
   - Each movie is represented as a numerical feature vector

3. **User Preference Modeling**
   - When a user selects multiple movies, their vectors are averaged to create a user preference vector

4. **Similarity Computation**
   - Cosine similarity is calculated between the user vector and all movie vectors
   - Selected movies are excluded from recommendations

5. **Final Scoring**
   - Similarity score is combined with normalized IMDb rating
   - A user-controlled slider adjusts the weighting between similarity and rating

6. **Ranking & Output**
   - Movies are ranked by final score
   - Top recommendations are displayed in an interactive table

---

## Running the Project Locally

### Prerequisites
- Python 3.8+

### Installation

```bash
pip install -r requirements.txt
