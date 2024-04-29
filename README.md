# AI lyric writer and dataset maker

# Spotify and Genius Lyrics Dataset Creation Tool

## Overview

This repository contains Python scripts for generating datasets that contain lyrics and other related data from Spotify and Genius. The purpose of this tool is to enable users to create custom datasets that could be used for various applications like data analysis, machine learning models, and more.

## Features

- **Authentication with Spotify and Genius**: Utilizes OAuth to authenticate and access Spotify and Genius APIs.
- **Dataset Creation**: Enables users to generate datasets from playlists, top songs of artists, or a specified list of artists.
- **Lyrics Extraction**: Fetches lyrics from the Genius website for the tracks obtained from Spotify.
- **Performance Monitoring**: Includes decorators for tracking the performance and operations of the dataset creation functions.

## Setup and Installation

1. **Clone the Repository**: First, clone this repository to your local machine using Git.

git clone https://github.com/MuhammetSonmez/AI-lyric-writer-and-dataset-maker/
```
  cd AI-lyric-writer-and-dataset-maker
```

2. **Install Required Packages**: Install the necessary Python packages using pip.
```
  pip install -r requirements.txt
```

3. **Environment Variables**: You need to set up environment variables for `CLIENT_ID`, `CLIENT_SECRET`, and `GENIUS_ACCESS_TOKEN` which you can obtain by registering your application with Spotify and Genius API platforms.
Create a `.env` file in the root directory and fill in your credentials:

```
  CLIENT_ID='your_spotify_client_id'
  CLIENT_SECRET='your_spotify_client_secret'
  GENIUS_ACCESS_TOKEN='your_genius_access_token'
```

## Usage

- **Creating a Dataset from a Spotify Playlist**:
Use the function `create_dataset_with_playlist` to create a dataset from a specific Spotify playlist. You need to pass the playlist ID and the desired dataset name.

```
  token = get_token()
  create_dataset_with_playlist(token, "playlist_id_here", "playlist_dataset")
```

- **Creating a Dataset with Top Ten Songs of an Artist**:
This function fetches the top ten songs of a given artist from Spotify and collects their lyrics from Genius.
```
  token = get_token()
  create_dataset_with_top_ten(token, "artist_name_here")
```

- **Creating a Dataset with a List of Artists**:
Generates datasets for multiple artists and combines them into one single dataset.

```
  artists = ["Aerosmith", "U2", "Bon Jovi"]
  create_dataset_with_artist_list(get_token(), "rock_dataset", artists)
```

# AI Lyrics Generation Model

## Overview

This Python script generates lyrics using a TensorFlow-based neural network. It employs character-level text generation to produce song lyrics in the style of the data it was trained on. The training data consists of song lyrics fetched from Spotify and Genius APIs.

## Features

- **Text Preprocessing**: Cleans and prepares text data for neural network training.
- **Model Training**: Uses a GRU (Gated Recurrent Unit) based architecture for learning dependencies in text.
- **Lyrics Generation**: Generates new lyrics based on a provided start string.
- **Loss Visualization**: Option to visualize training loss to assess model performance.

## Usage

### Train and Generate Lyrics
To train the model and generate lyrics, follow these steps:

1. **Prepare your Dataset**: Ensure you have a `.txt` file containing lyrics data, typically generated by the accompanying dataset creation script.

2. **Run the Script**:
   - Launch the script using Python. Training will commence using the data from the specified dataset path. After training, the model will attempt to generate lyrics starting from a predefined string.
   
3. **Generate Text**:
The main() function in the script accepts a start string and an optional boolean to draw the loss graph. Modify the start string as needed and set draw_loss to True if you want to visualize the training loss.

```
  if __name__ == '__main__':
      main("Start of your lyrics", True)
```


