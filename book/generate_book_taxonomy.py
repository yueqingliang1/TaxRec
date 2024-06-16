

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
directory = 'processed_data/'

books = pd.read_csv(directory + 'books.csv')


def get_taxonomy_sample(n_features):
    if n_features == 5:
        taxonomy = {
            # top-5
            "Genres": ["Fiction", "Non-Fiction", "Science Fiction", "Fantasy", "Mystery", "Thriller", "Romance", "Historical Fiction", "Biography", "Self-Help", "Children's", "Young Adult", "Humor", "Classics", "Poetry", "Graphic Novel", "Satire", "Urban Fantasy", "Dystopian"],
            "Themes": ["Adventure", "Friendship", "Love", "War", "Crime", "Fantasy", "Science", "Religion", "Philosophy", "Politics", "History", "Technology", "Magic", "Travel", "Art", "Music", "Sports", "Family", "Identity", "Growth", "Ethics", "Nature", "Survival", "Courage", "Betrayal", "Justice", "Freedom"],
            "Audience Age Groups": ["Children", "Teens", "Adults", "Seniors", "All Ages"],
            "Settings": ["Contemporary", "Historical", "Future", "Alternate Reality", "Urban", "Rural", "Space", "Fantasy World"],
            "Popularity": ["Bestseller", "Popular", "Average", "Niche"]
        }
        sample_output = {"Genres": ["Romance", "Thriller"], "Themes": ["Love", "Mystery", "Crime"], "Audience Age Groups": "Adults", "Settings": "Contemporary", "Popularity": "Popular"}
    
    elif n_features == 10:
        taxonomy = {
            # top-5
            "Genres": ["Fiction", "Non-Fiction", "Science Fiction", "Fantasy", "Mystery", "Thriller", "Romance", "Historical Fiction", "Biography", "Self-Help", "Children's", "Young Adult", "Humor", "Classics", "Poetry", "Graphic Novel", "Satire", "Urban Fantasy", "Dystopian"],
            "Themes": ["Adventure", "Friendship", "Love", "War", "Crime", "Fantasy", "Science", "Religion", "Philosophy", "Politics", "History", "Technology", "Magic", "Travel", "Art", "Music", "Sports", "Family", "Identity", "Growth", "Ethics", "Nature", "Survival", "Courage", "Betrayal", "Justice", "Freedom"],
            "Audience Age Groups": ["Children", "Teens", "Adults", "Seniors", "All Ages"],
            "Settings": ["Contemporary", "Historical", "Future", "Alternate Reality", "Urban", "Rural", "Space", "Fantasy World"],
            "Popularity": ["Bestseller", "Popular", "Average", "Niche"],

            # top-10
            "Languages": ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Italian", "Portuguese", "Russian", "Other"],
            "Formats": ["Hardcover", "Paperback", "Ebook", "Audiobook"],
            "Book Lengths": ["Short Story", "Novella", "Novel", "Series"],
            "Publication Years": ["Pre-1900", "1900-1950", "1951-2000", "2001-2010", "2011-2020", "2021-Present"],
            "Author Nationalities": ["American", "British", "Canadian", "Australian", "Indian", "Chinese", "Japanese", "French", "German", "Russian", "Italian", "Spanish", "Brazilian", "Mexican", "South African", "Haitian", "Other"]
        }
        sample_output = {"Genres": ["Romance", "Thriller"], "Themes": ["Love", "Mystery", "Crime"], "Audience Age Groups": "Adults", "Settings": "Contemporary", "Popularity": "Popular", "Languages": "English", "Formats": "Paperback", "Book Lengths": "Novel", "Publication Years": "1951-2000", "Author Nationalities": "American"}

    elif n_features == 15:
        taxonomy = {
            # top-5
            "Genres": ["Fiction", "Non-Fiction", "Science Fiction", "Fantasy", "Mystery", "Thriller", "Romance", "Historical Fiction", "Biography", "Self-Help", "Children's", "Young Adult", "Humor", "Classics", "Poetry", "Graphic Novel", "Satire", "Urban Fantasy", "Dystopian"],
            "Themes": ["Adventure", "Friendship", "Love", "War", "Crime", "Fantasy", "Science", "Religion", "Philosophy", "Politics", "History", "Technology", "Magic", "Travel", "Art", "Music", "Sports", "Family", "Identity", "Growth", "Ethics", "Nature", "Survival", "Courage", "Betrayal", "Justice", "Freedom"],
            "Audience Age Groups": ["Children", "Teens", "Adults", "Seniors", "All Ages"],
            "Settings": ["Contemporary", "Historical", "Future", "Alternate Reality", "Urban", "Rural", "Space", "Fantasy World"],
            "Popularity": ["Bestseller", "Popular", "Average", "Niche"],

            # top-10
            "Languages": ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Italian", "Portuguese", "Russian", "Other"],
            "Formats": ["Hardcover", "Paperback", "Ebook", "Audiobook"],
            "Book Lengths": ["Short Story", "Novella", "Novel", "Series"],
            "Publication Years": ["Pre-1900", "1900-1950", "1951-2000", "2001-2010", "2011-2020", "2021-Present"],
            "Author Nationalities": ["American", "British", "Canadian", "Australian", "Indian", "Chinese", "Japanese", "French", "German", "Russian", "Italian", "Spanish", "Brazilian", "Mexican", "South African", "Haitian", "Other"],
            
            # top-15
            "Author Genders": ["Male", "Female", "Non-Binary", "Other", "Unknown"],
            "Page Counts": ["Less than 100", "100-200", "201-300", "301-400", "401-500", "More than 500"],
            "Reading Levels": ["Beginner", "Intermediate", "Advanced"],
            "Awards": ["Pulitzer Prize", "Man Booker Prize", "National Book Award", "Hugo Award", "Nebula Award", "None"],
            "Ratings": ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars", "Unknown"]
        }

        sample_output = {"Genres": ["Romance", "Thriller"], "Themes": ["Love", "Mystery", "Crime"], "Audience Age Groups": "Adults", "Settings": "Contemporary", "Popularity": "Popular", "Languages": "English", "Formats": "Paperback", "Book Lengths": "Novel", "Publication Years": "1951-2000", "Author Nationalities": "American", "Author Genders": "Female", "Page Counts": "301-400", "Reading Levels": "Intermediate", "Awards": "None", "Ratings": "Unknown"}

    return taxonomy, sample_output

taxonomy, sample_output = get_taxonomy_sample(n_features)
sample_input = books.iloc[5087:5088].to_csv(index=False)

# gpt4
client = AzureOpenAI(
  api_key = "c6af48fe651d44bb80477d9f17918c3d",
  api_version = '2024-02-15-preview', # '2024-02-15-preview','2023-05-15'
  azure_endpoint = "https://gpt-35-1106.openai.azure.com"
)

book_taxonomy = dict()
for i in tqdm(range(len(books))):
    book = books.iloc[i:i+1].to_csv(index=False)
    book_name = books['book_title'][i]

    user_prompt = f"""
    <Instruction>
    You are a book classifier. Given a book, please classify it following the format of the given taxonomy.
    book taxonomy: {taxonomy}
    <End of Instruction>

    <Restrictions>
    You must fill out the value for each key in the taxonomy, if there is any feature you are not sure, fill it with ""Unknown"". Output the classification following the output format in the example below and don't give any explanation.
    <End of Restrictions>

    <Example>
    Input:
    {sample_input}

    Output:
    {sample_output}
    <End of Example>

    Input:
    {book}

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
    book_taxonomy[book_name] = output


with open('outputs/' + f'book_taxonomy_{n_features}feats_temp.json', 'w') as f:
    json.dump(book_taxonomy, f, indent=4)

