import dotenv
import os
import base64
import requests
import json
from bs4 import BeautifulSoup
import functools
import time

dotenv.load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv('CLIENT_SECRET')
genius_access_token = os.getenv('GENIUS_ACCESS_TOKEN')


def time_performance(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"time spend: {time.time() - start_time}")
        return result
    return wrapper

def operation_tracker(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Starting {func.__name__}...")
        result = func(*args, **kwargs)
        print(f"{func.__name__} completed with return value: {result}")
        return result
    return wrapper

def get_indexes(data:str, target:chr) -> list:
    indexes = []
    current_index = 0
    while True:
        current_index = data.find(target, current_index)
        if current_index == -1:
            break
        indexes.append(current_index)
        current_index += 1
    return indexes

def fix_data(data:str, target1:chr, target2:chr) -> str:

    x = get_indexes(data, target1)
    y = get_indexes(data, target2)
    if len(x) != len(y):
        return ""
        #raise IndexError("check your x and y lenght")
    scopes = [[x[i],y[i]] for i in range(len(x))]

    for scope in reversed(scopes):
        start, end = scope
        original_segment = data[start:end+1]
        modified_segment = original_segment.replace("\n", "")
        data = data[:start] + modified_segment + data[end+1:]

    return data

def get_token() -> str:
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode('utf-8')
    auth_base64 = str(base64.b64encode(auth_bytes), 'utf-8')
    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization":"Basic " + auth_base64,
        "Content-Type":"application/x-www-form-urlencoded"
    }
    data = {"grant_type":"client_credentials"}
    result = requests.post(url, headers=headers, data=data)
    json_result = json.loads(result.content)
    token = json_result["access_token"]
    return token



def get_auth_header(token:str) -> dict:
    return {"Authorization": "Bearer " + token}


@operation_tracker
def search_for_artist(token:str, artist_name:str) -> list:
    url = "https://api.spotify.com/v1/search?"
    headers = get_auth_header(token)
    query = f"q={artist_name}&type=artist&limit=1"

    query_url = url + query
    result = requests.get(query_url, headers=headers)
    json_result = json.loads(result.content)["artists"]["items"]
    if len(json_result) == 0:
        print("artist is not exists")
        return None
    return json_result[0]


@operation_tracker
def get_top_ten_songs_by_artists(token:str, artist_id:str) -> list:
    url = f"https://api.spotify.com/v1/artists/{artist_id}/top-tracks?country=US"
    headers = get_auth_header(token)
    result = requests.get(url, headers=headers)
    json_result = json.loads(result.content)["tracks"]
    return json_result


@operation_tracker
def get_song_url(song_title: str, artist_name: str) -> str:
    url = "https://api.genius.com/search"
    headers = {
        "Authorization": f"Bearer {genius_access_token}"
    }
    params = {
        "q": f"{song_title} {artist_name}"
    }
    response = requests.get(url, headers=headers, params=params)
    json_data = json.loads(response.content)
    try:
        song_info = json_data['response']['hits'][0]['result']
    except IndexError:
        return "Lyrics not found."
    
    if not song_info:
        return "Lyrics not found."
    song_api_path = song_info['api_path']

    song_url = "https://api.genius.com" + song_api_path
    song_response = requests.get(song_url, headers=headers)
    song_json = json.loads(song_response.content)
    
    return song_json["response"]["song"]["description_annotation"]["url"]



def parse_lyrics(url:str) -> str:
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    lyrics_divs = soup.find_all('div', class_='Lyrics__Container-sc-1ynbvzw-1 kUgSbL')    
    lyrics = []
    for div in lyrics_divs:
        lyrics.append(div.get_text(separator='\n'))
    return lyrics if lyrics else "Lyrics not found."


@operation_tracker
def get_lyrics(songs:str, artist:str, i:int, json_flag:bool = True) -> str:
    result = ""
    if json_flag:
        song_url = get_song_url(songs[i]["name"], artist)
    else:
        song_url = get_song_url(songs, artist)

    if song_url == "Lyrics not found.":
        return ""
    lyrics = parse_lyrics(song_url)

    if isinstance(lyrics, list):
        for lyric in lyrics:
            result += lyric
    result = fix_data(result, "[", "]")
    result = fix_data(result, "(", ")")

    return result + "\n"



@operation_tracker
def extract_track_names(tracks: list) -> list:
    track_details = [{
        'track_name': track['track']['name'],
        'artists': [artist['name'] for artist in track['track']['artists']]
        } for track in tracks if track['track']]
    return track_details

@operation_tracker
def get_playlist_tracks(token: str, playlist_id: str) -> list:
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    headers = get_auth_header(token)
    tracks = []
    while url:
        response = requests.get(url, headers=headers)
        json_response = json.loads(response.content)
        tracks.extend(json_response['items'])
        url = json_response.get('next')
    tracks = extract_track_names(tracks)
    return tracks


#@operation_tracker
@time_performance
def create_dataset_with_playlist(token:str, playlist_id:str, dataset_name:str) -> None:
    data = ""
    tracks = get_playlist_tracks(token, playlist_id)
    i = 0
    for track in tracks:
        track_name = track['track_name']
        artist = track['artists'][0]
        data += get_lyrics(track_name, artist, i, json_flag=False)        
        i += 1
    
    with open(fr"datasets\{dataset_name}-dataset.txt", "w", encoding="utf-8") as f:
        f.write(data)
        f.close()    




    """
    create_dataset_with_playlist(get_token(), "0GfqXlle567tqjdfVZvOJI", "main-dataset")
    """



# @operation_tracker
def create_dataset_with_top_ten(token:str, artist:str) -> None:

    result = search_for_artist(token, artist)
    try:
        artist_id = result["id"]
    except UnboundLocalError:
        print("artist does not exist")
        exit()
    songs = get_top_ten_songs_by_artists(token, artist_id)

    data = ""
    for i in range(len(songs)):
        data += get_lyrics(songs, artist, i)
    

    with open(fr"datasets\{artist.lower().replace('/', '-')}-dataset.txt", "w", encoding="utf-8") as f:
        f.write(data)
        f.close()
    
    """
    token = get_token()
    artist = "The Beatles"
    create_dataset_with_top_ten(token, artist)

    """


@time_performance
def create_dataset_with_artist_list(token:str, datasetName:str, artists:list) -> None:
    
    # creating small datasets
    for artist in artists:
        create_dataset_with_top_ten(token, artist)
    
    
    dataset = ""
    for artist in artists:
        with open(fr"datasets\{artist.lower().replace('/', '-')}-dataset.txt", "r", encoding="utf-8") as f:
            dataset += f.read() + "\n"
            print(dataset)
            f.close()
    
    with open(fr"datasets\{datasetName.lower()}-dataset.txt", "+a", encoding="utf-8") as f:
        f.write(dataset)
        f.close()
    """
    create_dataset_with_artist_list(get_token(),datasetName="rock-test", artists = ["Aerosmith", "U2","Bon Jovi",
                                        "Red Hot Chili Peppers","Radiohead",
                                        "Linkin Park"])
    """


create_dataset_with_playlist(get_token(), "0GfqXlle567tqjdfVZvOJI", "main-dataset")
