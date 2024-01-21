import streamlit as st
import pickle
import pandas as pd
from sklearn.cluster import KMeans
import random
import joblib

st.set_page_config(
        page_title="About Project",
        page_icon="🤔",
    )

st.title("About Project")

st.markdown("""
            This project was made in Python, using various packages, including Streamlit, Pandas, Scikit-learn, and Joblib. The dataset used in this project was obtained from Kaggle, and it contains over 160,000 tracks from Spotify's API. The dataset was preprocessed using Python, and the K-Means clustering algorithm was used to cluster the tracks into 11 different genres. The K-Means model was then saved using Joblib, and the web app was created using Streamlit.

            """)

st.write("## Dataset")

st.image('dataset.png',caption="Dataset of tracks from Kaggle")

st.write("The dataset was acquired from Kaggle, made by user maharshipandya. The dataset can be seen [here](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset).")

st.write("## Histogram")

st.image('histogram.png',caption="Histograms of various audio features within a dataset of tracks")

st.write("The image represents histograms of various audio features within a dataset of tracks, showcasing the data after preprocessing. Each histogram illustrates the distribution of a specific feature, such as acousticness, danceability, energy, instrumentalness, liveness, speechiness, tempo, valence, and loudness. Histograms serve as a visual summary of the dataset's characteristics, revealing key insights. For instance, the range of values for each feature (distribution shape) and other relevant information. These processed histograms are instrumental for subsequent data analysis or machine learning tasks, providing a foundation for more in-depth exploration and modeling.")

st.write("## Heatmap")

st.image('heatmap.png', caption="Heatmap of the correlation between audio features")

st.write("## Clustering")

st.write("The graphs below show the clustering of tracks into 11 different genres and KMeans labels. The clustering was done using the K-Means algorithm. ")

st.image('pca_scatter_genres.png',caption="Clustering of tracks into 11 different genres")
st.image('pca_scatter_KM.png', caption="Clustering of tracks into 11 different KMeans labels")


st.write("The cluster labels are as folows:")

st.markdown("""
    - KM0: "Happy & Upbeat"
    - KM1: "Chill Acoustic Groove"
    - KM2: "Versatile Instrumental Vibes"
    - KM3: "Smooth and Subtle"
    - KM4: "Energetic Positivity"
    - KM5: "Soothing Acoustic Bliss"
    - KM6: “Harmonious Instrumental Oasis"
    - KM7:” Enchanting Acoustic Harmony"
    - KM8: "Feel-Good Energy"
    - KM9: "Lively Dance Serenade"
    - KM10: “Dance-Powered Energy"
            """)
