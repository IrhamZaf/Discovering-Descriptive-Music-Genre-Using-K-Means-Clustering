import streamlit as st
import pickle
import pandas as pd
from sklearn.cluster import KMeans
import random
import joblib

st.set_page_config(
        page_title="Discovering Descriptive Music Genre Using K-Means Clustering",
        page_icon="🎵",
    )

df = pd.read_csv('MusicNormalized.csv')


# Load the KMeans model from the pickle file
kmeans = joblib.load('trained_model.joblib')

st.write("# Discovering Descriptive Music Genre Using K-Means Clustering")

st.write("Welcome to the Discovering Descriptive Music Genre Using K-Means Clustering web app! This is a final year project made by Muhamad Irham Zafran in partial requirement for the degree of Bachelor of Information Systems (Hons.) Intelligent Systems Engineering.")

st.sidebar.success("Please navigate through the sidebar.")

# # Display the KMeans labels and song list
# st.write("KMeans Labels:")
# for label in filtered_df['KMeansLabel'].unique():
#     st.write(label)








# filtered_df['KMeansLabel'] = kmeans.predict(filtered_df[audio_cols])

# # Display the KMeans labels and song list
# st.write("KMeans Labels:")
# for label in filtered_df['KMeansLabel'].unique():
#     st.write(label)

# st.write("\nSong List:")
# for index, row in filtered_df.iterrows():
#     st.write(f"{row['track_name']} - {row['artists']} - {row['genre'] }")


# Apply the KMeans algorithm to the filtered data
# filtered_df = df.copy()
# for col in audio_cols:
#     if options[col] == 'low':
#         filtered_df = filtered_df[filtered_df[col] < random.uniform(0.0, 0.3)]
#     elif options[col] == 'mid':
#         filtered_df = filtered_df[filtered_df[col] > random.uniform(0.3, 0.5)]
#     else:
#         filtered_df = filtered_df[filtered_df[col] > random.uniform(0.5, 0.7)]

# # Print the values of filtered_df
# st.write("\nFiltered Data:")
# st.write(filtered_df)

# filtered_df['KMeansLabel'] = kmeans.predict(filtered_df[audio_cols])

# # Display the KMeans labels and song list
# st.write("KMeans Labels:")
# for label in filtered_df['KMeansLabel'].unique():
#     st.write(label)

# st.write("\nSong List:")
# for index, row in filtered_df.iterrows():
#     st.write(f"{row['track_name']} - {row['artists']} - {row['genre'] }")


