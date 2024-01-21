import streamlit as st
import pickle
import pandas as pd
from sklearn.cluster import KMeans
import random
import joblib

st.set_page_config(
        page_title="Project Demo",
        page_icon="🧑‍💻",
    )

df = pd.read_csv('MusicNormalized.csv')


# Load the KMeans model from the pickle file
kmeans = joblib.load('trained_model.joblib')

st.title("Project Demo")



st.write("Select your preferred music attributes:")

# Define the song attributes
audio_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence', 'loudness']

# Create the radio buttons for each song attribute
options = {}
for col in audio_cols:
    options[col] = st.radio(col, ('low', 'mid', 'high'))

# Apply the KMeans algorithm to the filtered data
filtered_df = df.copy()
values = []
for col in audio_cols:
    if options[col] == 'low':
        value = random.uniform(0.0, 0.3)
    elif options[col] == 'mid':
        value = random.uniform(0.3, 0.5)
    elif options[col] == 'high':
        value = random.uniform(0.5, 0.7)
    values.append(value)


prediction = kmeans.predict([values])

filtered_df['KMeansLabel'] = kmeans.predict(filtered_df[audio_cols])

# st.write(prediction)

# Display the KMeans labels and song list

descriptive_labels = ['Happy & Upbeat',
                      'Chill Acoustic Groove',
                      'Versatile Instrumental Vibes',
                      'Smooth and Subtle',
                      'Energetic Positivity',
                      'Soothing Acoustic Bliss',
                      'Harmonious Instrumental Oasis',
                      'Enchanting Acoustic Harmony',
                      'Feel-Good Energy',
                      'Lively Dance Serenade',
                      'Dance-Powered Energy']


st.write("You seem to like songs that are:")
for label in prediction:
    st.write(descriptive_labels[label])
    # st.write(label)

    # filter so the filtered_df only contains the songs with the same label as the prediction

    filtered_df = filtered_df[filtered_df['KMeansLabel'] == label]

    st.write("You may like these songs:")
    # st.write(filtered_df)

    # show it in table form, however only display the track name, artists, and genre, and also its kmeans label

    # st.write("\nSong List:")
    # for index, row in filtered_df.iterrows():
    #     st.write(f"{row['track_name']} - {row['artists']} - {row['genre'] }")

    # show it in a table
        
    st.write(filtered_df[['track_name', 'artists', 'genre', 'KMeansLabel']])
