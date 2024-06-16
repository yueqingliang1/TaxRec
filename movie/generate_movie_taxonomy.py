

import json
import pandas as pd
from openai import AzureOpenAI
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_features', type=int, default=20, help='Number of features in the taxonomy')
args = parser.parse_args()
n_features = args.n_features


# Load the dataset
directory = 'ml-100k/'
f = open(directory + 'u.data', 'r')
data = f.readlines()
f = open(directory + 'u.item', 'r', encoding='ISO-8859-1')
movies = f.readlines()


movie_names = [_.split('|')[1] for _ in movies] # movie_names[0] = 'Toy Story (1995)'
movie_ids = [_.split('|')[0] for _ in movies] # movie_ids[0] = '1'
movie_dict = dict(zip(movie_ids, movie_names))
interaction_dicts = dict()  
for line in data:
    user_id, movie_id, rating, timestamp = line.split('\t')
    if user_id not in interaction_dicts:
        interaction_dicts[user_id] = {
            'movie_id': [],
            'rating': [],
            'timestamp': [],
            'movie_title': [],
        }
    interaction_dicts[user_id]['movie_id'].append(movie_id)
    interaction_dicts[user_id]['rating'].append(int(int(rating) > 3))
    interaction_dicts[user_id]['timestamp'].append(timestamp)
    interaction_dicts[user_id]['movie_title'].append(movie_dict[movie_id])

def get_taxonomy_sample(n_features):
    if n_features == 5:
        taxonomy = {
            "Genre": ["Action", "Comedy", "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Science Fiction", "Thriller", "Documentary", "Animation", "Biographical", "Musical", "Experimental", "Indie", "Western", "Historical"],
            "Themes": ["Love", "Adventure", "Friendship", "Survival", "Betrayal", "Justice", "Existential", "Philosophical", "Political", "Social Commentary"],
            "Streaming Availability": ["Netflix", "Amazon Prime", "Hulu", "Disney+", "HBO Max", "Apple TV+", "Other", "Not Available"],
            "IMDb": ["Below 6", "6+", "7+", "8+", "9+", "Not Applicable"],
            "Rotten Tomatoes": ["Below 60%", "60%+", "70%+", "80%+", "90%+", "Not Applicable"],
        }
        sample_output = {"Genre": "Documentary", "Themes": "Unknown", "Streaming Availability": "Unknown", "IMDb": "8+", "Rotten Tomatoes": "90%"}
    elif n_features == 10:
        taxonomy = {
            "Genre": ["Action", "Comedy", "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Science Fiction", "Thriller", "Documentary", "Animation", "Biographical", "Musical", "Experimental", "Indie", "Western", "Historical"],
            "Themes": ["Love", "Adventure", "Friendship", "Survival", "Betrayal", "Justice", "Existential", "Philosophical", "Political", "Social Commentary"],
            "Streaming Availability": ["Netflix", "Amazon Prime", "Hulu", "Disney+", "HBO Max", "Apple TV+", "Other", "Not Available"],
            "IMDb": ["Below 6", "6+", "7+", "8+", "9+", "Not Applicable"],
            "Rotten Tomatoes": ["Below 60%", "60%+", "70%+", "80%+", "90%+", "Not Applicable"],
            "Metacritic": ["Below 50", "50+", "60+", "70+", "80+", "90+", "Not Applicable"],
            "MPAA Rating": ["G", "PG", "PG-13", "R", "NC-17", "Not Rated", "Unrated"],
            "Language": ["English", "Spanish", "French", "Chinese", "Japanese", "Korean", "Other"],
            "Release Year": ["Classic (Pre-1980)", "1980s", "1990s", "2000s", "2010s", "Recent (Last 5 years)", "Current Year"],
            "Critical Reception": ["Poorly Received", "Mixed Reviews", "Critically Acclaimed", "Award-Winning"],
        }
        sample_output = {"Genre": "Documentary", "Themes": "Unknown", "Streaming Availability": "Unknown", "IMDb": "8+", "Rotten Tomatoes": "90%", "Metacritic": "90+", "MPAA Rating": "R", "Critical Reception": "Critically Acclaimed"}
    elif n_features == 20:
        taxonomy = {
            # top-5
            "Genre": ["Action", "Comedy", "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Science Fiction", "Thriller", "Documentary", "Animation", "Biographical", "Musical", "Experimental", "Indie", "Western", "Historical"],
            "Themes": ["Love", "Adventure", "Friendship", "Survival", "Betrayal", "Justice", "Existential", "Philosophical", "Political", "Social Commentary"],
            "Streaming Availability": ["Netflix", "Amazon Prime", "Hulu", "Disney+", "HBO Max", "Apple TV+", "Other", "Not Available"],
            "IMDb": ["Below 6", "6+", "7+", "8+", "9+", "Not Applicable"],
            "Rotten Tomatoes": ["Below 60%", "60%+", "70%+", "80%+", "90%+", "Not Applicable"],

            # top-10
            "Metacritic": ["Below 50", "50+", "60+", "70+", "80+", "90+", "Not Applicable"],
            "MPAA Rating": ["G", "PG", "PG-13", "R", "NC-17", "Not Rated", "Unrated"],
            "Language": ["English", "Spanish", "French", "Chinese", "Japanese", "Korean", "Other"],
            "Release Year": ["Classic (Pre-1980)", "1980s", "1990s", "2000s", "2010s", "Recent (Last 5 years)", "Current Year"],
            "Critical Reception": ["Poorly Received", "Mixed Reviews", "Critically Acclaimed", "Award-Winning"],

            # top-20
            "Country of Origin": ["United States", "United Kingdom", "France", "India", "China", "Japan", "South Korea", "Italy", "Spain", "Germany", "Other"],
            "Runtime": ["Short (< 90 minutes)", "Medium (90-120 minutes)", "Long (> 120 minutes)", "Variable (Anthology or Series)"],
            "Awards": ["Oscar", "Golden Globe", "Cannes", "BAFTA", "Sundance", "Berlin International Film Festival", "Venice Film Festival", "Other"],
            "Visual Style": ["Live Action", "Animation", "Stop Motion", "Mixed Media", "Black and White", "Color"],
            "Technology": ["2D", "3D", "IMAX", "VR", "Dolby Atmos", "4DX"],
            "Adaptation": ["Original", "Based on a Book", "Based on True Events", "Remake", "Sequel", "Franchise"],
            "Content Warning": ["None", "Violence", "Sexual Content", "Language", "Drug Use"],
            "Budget": ["Low Budget", "Medium Budget", "High Budget", "Blockbuster"],
            "Box Office Revenue": ["Flop", "Below Expectation", "Moderate", "Successful", "Blockbuster"],
            "Cultural Impact": ["Cult Classic", "Iconic", "Influential", "Controversial", "Negligible"]
        }
        sample_output = {"Genre": "Documentary", "Themes": "Unknown", "Streaming Availability": "Unknown", "IMDb": "8+", "Rotten Tomatoes": "90%+", "Metacritic": "90+", "MPAA Rating": "R", "Critical Reception": "Critically Acclaimed", "Runtime": "Medium (90-120 minutes)", "Visual Style": "Unknown", "Technology": "Unknown", "Language": "English", "Country of Origin": "United States", "Release Year": "1990s", "Awards": "Unknown", "Budget": "Unknown", "Box Office Revenue": "Unknown", "Cultural Impact": "Unknown", "Adaptation": "Original", "Content Warning": "Unknown"}
    return taxonomy, sample_output

taxonomy, sample_output = get_taxonomy_sample(n_features)

# gpt4
client = AzureOpenAI(
  api_key = "<your key>",
  api_version = '2024-02-15-preview', # '2024-02-15-preview','2023-05-15'
  azure_endpoint = "<your endpoint>"
)

movie_taxonomy = dict()
for movie in tqdm(movie_names):
    # instruction = f"{taxonomy}\nGiven a movie, classify it into the taxonomy. Output the classification in json format and don't give any explanation."
    user_prompt = f"""
    <Instruction>
    You are a movie classifier. Given a movie, please classify it following the format of the given taxonomy.
    Movie taxonomy: {taxonomy}
    <End of Instruction>

    <Restrictions>
    You must fill out the value for each key in the taxonomy, if there is any feature you are not sure, fill it with ""Unknown"". Output the classification in json format and don't give any explanation.
    <End of Restrictions>

    <Example>
    Input:
    Crumb (1994)

    Output:
    {sample_output}
    <End of Example>

    Input:
    {movie}

    Output:

    """
    
    response = client.chat.completions.create(
        model="GPT4-WEST-US",
        temperature=0.2,
        messages=[
            # {"role": "system", "content": instruction},
            {"role": "user", "content": user_prompt}
        ]
    )
    output = response.choices[0].message.content
    movie_taxonomy[movie] = output


with open(directory + f'movie_taxonomy_{n_features}feats_temp.json', 'w') as f:
    json.dump(movie_taxonomy, f, indent=4)

