# environment: conda activate llm
# nohup python chatgpt4_tax.py > exp_log/chatgpt4_tax.txt 2>&1 &

import json
from openai import AzureOpenAI
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_features', type=int, default=20, help='Number of features in the taxonomy. n_features=[5,10,20]')
parser.add_argument('--k', type=int, default=10, help='Number of generated recommendations. Recall@k. k=[1,5,10]')
parser.add_argument('--output_path', type=str, default='valid_2000_tax_4_gpt4.json', help='Output file path')
args = parser.parse_args()

n_features = args.n_features
k = args.k
output_path = args.output_path

if n_features == 5:
    path = 'valid_2000_tax_5feats.json'
elif n_features == 10:
    path = 'valid_2000_tax_10feats.json'
elif n_features == 20:
    path = 'valid_2000_tax_20feats.json'

# Load the JSON dataset
directory = 'ml-100k/'

with open(directory + path, 'r') as file:
    data = json.load(file)


# Initialize the OpenAI API client
client = AzureOpenAI(
  api_key = "<your key>",
  api_version = '2024-02-15-preview',
  azure_endpoint = "<your endpoint>"
)


def generate_prompt_sample(taxonomy, k):
    import random
    sample_recommendations = []
    for _ in range(k):
        recommendation = []
        for category, options in taxonomy.items():
            recommendation.append([random.choice(options)])
        sample_recommendations.append(recommendation)
    return sample_recommendations

def extract_movie_features(taxonomy):
    movie_features = []
    for category, options in taxonomy.items():
        movie_features.append(category)
    return movie_features



def generate_prompt_samples(n_features, k):
    if n_features == 5:
        watching_history = "\"Primal Fear (1996): ['Thriller', 'Justice', 'Unknown', '7+', '70%+']\", \"Truth About Cats & Dogs, The (1996): ['Romance', 'Love', 'Unknown', '6+', '80%+']\", \"Lone Star (1996): ['Drama', 'Mystery', 'Unknown', '7+', '80%+']\", \"Last Supper, The (1995): ['Drama', 'Social Commentary', 'Unknown', '7+', 'Unknown']\", \"Jerry Maguire (1996): ['Comedy', 'Love', 'Unknown', '7+', '80%+']\", \"Crucible, The (1996): ['Drama', 'Justice', 'Unknown', '7+', '70%+']\", \"Leaving Las Vegas (1995): ['Drama', 'Love', 'Unknown', '7+', '90%+']\", \"Frighteners, The (1996): ['Horror', 'Comedy', 'Unknown', '7+', '60%+']\", \"Long Kiss Goodnight, The (1996): ['Action', 'Betrayal', 'Unknown', '6+', '60%+']\", \"Birdcage, The (1996): ['Comedy', 'Social Commentary', 'Unknown', '7+', '80%+']\""
        if k == 1:
            example_output = "{\n  \"L.A. Confidential (1997)\": ['Mystery', 'Betrayal', 'Unknown', '8+', '90%+']\n}"
        # elif k == 5:
        #     example_output = "{\n  \"recommendations\": \"[\"L.A. Confidential (1997)\",\"Good Will Hunting (1997)\",\"Fargo (1996)\",\"Trainspotting (1996)\",\"English Patient, The (1996)\"]\"\n}"
        # elif k == 10:
        #     example_output = "{\n  \"recommendations\": \"[\"L.A. Confidential (1997)\",\"Good Will Hunting (1997)\",\"Fargo (1996)\",\"Trainspotting (1996)\",\"English Patient, The (1996)\",\"Boogie Nights (1997)\",\"Seven (Se7en) (1995)\",\"Heat (1995)\",\"Shine (1996)\",\"Sling Blade (1996)\"]\"\n}"
    # elif n_features == 10:
    #     if k == 1:
    #         example_output = "{\n  \"recommendations\": \"[\"L.A. Confidential (1997)\"]\"\n}"
        # elif k == 5:
        #     example_output = "{\n  \"recommendations\": \"[\"L.A. Confidential (1997)\",\"Good Will Hunting (1997)\",\"Fargo (1996)\",\"Trainspotting (1996)\",\"English Patient, The (1996)\"]\"\n}"
        # elif k == 10:
        #     example_output = "{\n  \"recommendations\": \"[\"L.A. Confidential (1997)\",\"Good Will Hunting (1997)\",\"Fargo (1996)\",\"Trainspotting (1996)\",\"English Patient, The (1996)\",\"Boogie Nights (1997)\",\"Seven (Se7en) (1995)\",\"Heat (1995)\",\"Shine (1996)\",\"Sling Blade (1996)\"]\"\n}"
    elif n_features == 20:
        watching_history = "\"Primal Fear (1996): ['Thriller', 'Justice', 'Unknown', '7+', '70%+', '60+', 'R', 'Critically Acclaimed', 'Long (> 120 minutes)', 'Live Action', '2D', 'English', 'United States', '1990s', 'Other', 'Unknown', 'Successful', 'Unknown', 'Based on a Book', 'Violence']\", \"Truth About Cats & Dogs, The (1996): ['Romance', 'Love', 'Unknown', '6+', '80%+', 'Unknown', 'PG-13', 'Mixed Reviews', 'Medium (90-120 minutes)', 'Color', '2D', 'English', 'United States', '1990s', 'Unknown', 'Unknown', 'Moderate', 'Negligible', 'Original', 'Sexual Content']\", \"Lone Star (1996): ['Drama', 'Mystery', 'Unknown', '7+', '80%+', '70+', 'R', 'Critically Acclaimed', 'Long (> 120 minutes)', 'Color', '2D', 'English', 'United States', '1990s', 'Other', 'Unknown', 'Moderate', 'Negligible', 'Original', 'Language']\", \"Last Supper, The (1995): ['Drama', 'Social Commentary', 'Unknown', '7+', 'Unknown', 'Unknown', 'R', 'Unknown', 'Medium (90-120 minutes)', 'Unknown', 'Unknown', 'English', 'United States', '1990s', 'Unknown', 'Unknown', 'Unknown', 'Unknown', 'Original', 'Unknown']\", \"Jerry Maguire (1996): ['Comedy', 'Love', 'Unknown', '7+', '80%+', '70+', 'R', 'Critically Acclaimed', 'Long (> 120 minutes)', 'Live Action', '2D', 'English', 'United States', '1990s', 'Oscar', 'High Budget', 'Successful', 'Iconic', 'Original', 'Language']\", \"Crucible, The (1996): ['Drama', 'Justice', 'Unknown', '7+', '70%+', '60+', 'PG-13', 'Mixed Reviews', 'Long (> 120 minutes)', 'Live Action', '2D', 'English', 'United States', '1990s', 'Other', 'Unknown', 'Below Expectation', 'Negligible', 'Based on a Book', 'Unknown']\", \"Leaving Las Vegas (1995): ['Drama', 'Love', 'Unknown', '7+', '90%+', '80+', 'R', 'Critically Acclaimed', 'Medium (90-120 minutes)', 'Live Action', '2D', 'English', 'United States', '1990s', 'Oscar', 'Low Budget', 'Moderate', 'Negligible', 'Based on a Book', 'Sexual Content']\", \"Frighteners, The (1996): ['Horror', 'Comedy', 'Unknown', '7+', '60%+', 'Unknown', 'R', 'Mixed Reviews', 'Medium (90-120 minutes)', 'Live Action', 'Unknown', 'English', 'New Zealand', '1990s', 'Unknown', 'Medium Budget', 'Below Expectation', 'Negligible', 'Original', 'Violence']\", \"Long Kiss Goodnight, The (1996): ['Action', 'Betrayal', 'Unknown', '6+', '60%+', 'Unknown', 'R', 'Mixed Reviews', 'Long (> 120 minutes)', 'Live Action', '2D', 'English', 'United States', '1990s', 'Unknown', 'Unknown', 'Moderate', 'Negligible', 'Original', 'Violence']\", \"Birdcage, The (1996): ['Comedy', 'Social Commentary', 'Unknown', '7+', '80%+', '60+', 'R', 'Mixed Reviews', 'Medium (90-120 minutes)', 'Live Action', '2D', 'English', 'United States', '1990s', 'Unknown', 'Successful', 'Unknown', 'Remake', 'Sexual Content']\""
        if k == 1:
            example_output = "{\n  \"L.A. Confidential (1997)\": ['Mystery', 'Betrayal', 'Unknown', '8+', '90%+', '80+', 'R', 'Critically Acclaimed', 'Long (> 120 minutes)', 'Color', '2D', 'English', 'United States', '1990s', 'Oscar', 'Medium Budget', 'Successful', 'Iconic', 'Based on a Book', 'Violence']\n}"
    return watching_history, example_output



if n_features == 5:
    taxonomy = {
        "Genre": ["Action", "Comedy", "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Science Fiction", "Thriller", "Documentary", "Animation", "Biographical", "Musical", "Experimental", "Indie", "Western", "Historical"],
        "Themes": ["Love", "Adventure", "Friendship", "Survival", "Betrayal", "Justice", "Existential", "Philosophical", "Political", "Social Commentary", "Unknown"],
        "Streaming Availability": ["Netflix", "Amazon Prime", "Hulu", "Disney+", "HBO Max", "Apple TV+", "Other", "Unknown"],
        "IMDb": ["Below 6", "6+", "7+", "8+", "9+", "Unknown"],
        "Rotten Tomatoes": ["Below 60%", "60%+", "70%+", "80%+", "90%+", "Unknown"],
    }

    watching_history, sample_recommendations = generate_prompt_samples(n_features, k)


    prompt = f"""
    <Instruction>
    You are a movie recommender system. Given a list of movies the user has watched before, please recommend {k} movies in a list of features following the format of the given taxonomy.
    Movie taxonomy: {taxonomy}
    The input and output format of each movie is represented by a list of features. 
    Given the movie taxonomy and a list of movies the user has watched before, please recommend {k} new types of movies that aligns with the user's viewing preferences.
    <End of Instruction>

    <Restrictions>
    Output the recommended movie with their type in a list of features:
    [[Genre], [Themes], [Streaming Availability], [IMDb], [Rotten Tomatoes]]
    Fill each feature in '[]' with the corresponding value based on the movie_taxonomy. The output only consists of the movie name and a list of features without any other texts.",
    <End of Restrictions>

    <Example>
    Input:
    The user has watched the following movies before:{watching_history}\n 
    Output:
    {sample_recommendations}
    <End of Example>

    Input:
    """

elif n_features == 20:
    taxonomy = {
        # top-5
        "Genre": ["Action", "Comedy", "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Science Fiction", "Thriller", "Documentary", "Animation", "Biographical", "Musical", "Experimental", "Indie", "Western", "Historical"],
        "Themes": ["Love", "Adventure", "Friendship", "Survival", "Betrayal", "Justice", "Existential", "Philosophical", "Political", "Social Commentary", "Unknown"],
        "Streaming Availability": ["Netflix", "Amazon Prime", "Hulu", "Disney+", "HBO Max", "Apple TV+", "Other", "Unknown"],
        "IMDb": ["Below 6", "6+", "7+", "8+", "9+", "Unknown"],
        "Rotten Tomatoes": ["Below 60%", "60%+", "70%+", "80%+", "90%+", "Unknown"],

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
    # sample_recommendations = generate_prompt_sample(taxonomy, k)
    watching_history, sample_recommendations = generate_prompt_samples(n_features, k)

    prompt = f"""
    <Instruction>
    You are a movie recommender system. Given a list of movies the user has watched before, please recommend {k} movies in a list of features following the format of the given taxonomy.
    Movie taxonomy: {taxonomy}
    The input and output format of each movie is represented by a list of features. 
    Given the movie taxonomy and a list of movies the user has watched before, please recommend {k} new types of movies that aligns with the user's viewing preferences.
    <End of Instruction>

    <Restrictions>
    Output the recommended movie with their type in a list of features:
    [[Genre], [Themes], [Streaming Availability], [IMDb], [Rotten Tomatoes]]
    Fill each feature in '[]' with the corresponding value based on the movie_taxonomy. The output only consists of the movie name and a list of features without any other texts.",
    <End of Restrictions>

    <Example>
    Input:
    The user has watched the following movies before:{watching_history}\n 
    Output:
    {sample_recommendations}
    <End of Example>

    Input:
    """

for entry in tqdm(data):
    user_prompt = prompt + entry["input"] + "\nOutput:"
    response = client.chat.completions.create(
        model="GPT4-WEST-US",
        temperature=0.2,
        messages=[
            # {"role": "system", "content": entry["instruction"]},
            {"role": "user", "content": user_prompt}
        ]
    )

    output = response.choices[0].message.content
    entry["output"] = output

with open(directory + output_path, 'w') as file:
    json.dump(data, file, indent=2)
