# environment: conda activate llm
# nohup python chatgpt4_tax.py > exp_log/chatgpt4_tax.txt 2>&1 &

import json
from openai import AzureOpenAI
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_features', type=int, default=10, help='Number of features in the taxonomy. n_features=[5,10,20]')
parser.add_argument('--k', type=int, default=1, help='Number of generated recommendations. Recall@k. k=[1,5,10]')
parser.add_argument('--mode', type=str, default='tax_tax', help='Prompt mode: nametax_nametax, nametax_tax, tax_tax')
parser.add_argument('--output_path', type=str, default='data_2000_tax_5feats_name_k1_gpt4.json', help='Output file path')
args = parser.parse_args()

n_features = args.n_features
k = args.k
mode = args.mode
output_path = args.output_path

if mode == 'tax_tax':
    if n_features == 5:
        # path = 'data_2000_nametax_5feats.json'
        path = 'data_2000_tax_5feats.json'
    elif n_features == 10:
        # path = 'data_2000_nametax_10feats.json'
        path = 'data_2000_tax_10feats.json'
    elif n_features == 15:
        # path = 'data_2000_nametax_15feats.json'
        path = 'data_2000_tax_15feats.json'
else:
    if n_features == 5:
        path = 'data_2000_nametax_5feats.json'
        # path = 'data_2000_tax_5feats.json'
    elif n_features == 10:
        path = 'data_2000_nametax_10feats.json'
        # path = 'data_2000_tax_10feats.json'
    elif n_features == 15:
        path = 'data_2000_nametax_15feats.json'
        # path = 'data_2000_tax_15feats.json'

# Load the JSON dataset
input_dir = 'processed_data/'
with open(input_dir + path, 'r') as file:
    data = json.load(file)



def extract_book_features(taxonomy):
    book_features = []
    for category, options in taxonomy.items():
        book_features.append(category)
    return book_features



def generate_prompt_samples_nametax_nametax(n_features, k):
    if n_features == 5:
        interaction_history = "{\"Patriot Games\": [['Thriller', 'Fiction'], ['Adventure', 'War', 'Politics', 'Crime'], 'Adults', 'Contemporary', 'Bestseller'], \"The Pillars of the Earth\": [['Historical Fiction'], ['Adventure', 'Love', 'War', 'Politics', 'History'], 'Adults', 'Historical', 'Bestseller'], \"The Firm\": [['Thriller', 'Mystery'], ['Crime', 'Justice', 'Betrayal'], 'Adults', 'Contemporary', 'Bestseller'], \"Insomnia\": [['Horror', 'Fantasy'], ['Psychological', 'Supernatural', 'Mystery'], 'Adults', 'Contemporary', 'Bestseller'], \"Second Child\": [['Thriller', 'Horror'], ['Crime', 'Family', 'Mystery'], 'Adults', 'Contemporary', 'Popular'], \"Skeleton Crew\": [['Fiction', 'Horror', 'Thriller'], ['Survival', 'Horror', 'Mystery'], 'Adults', 'Contemporary', 'Bestseller'], \"The Stand: The Complete &amp; Uncut Edition\": [['Science Fiction', 'Horror', 'Fantasy'], ['Survival', 'Good vs. Evil', 'Apocalyptic'], 'Adults', 'Contemporary', 'Bestseller'], \"The Runaway Jury\": [['Thriller', 'Mystery'], ['Crime', 'Justice', 'Betrayal'], 'Adults', 'Contemporary', 'Bestseller'], \"The Client\": [['Thriller', 'Mystery'], ['Crime', 'Justice', 'Survival'], 'Adults', 'Contemporary', 'Bestseller'], \"The Pelican Brief\": [['Thriller', 'Mystery'], ['Crime', 'Justice', 'Politics'], 'Adults', 'Contemporary', 'Bestseller']}"
        if k == 1:
            example_output = "{\n  \"Nathaniel\": [['Horror', 'Thriller'], ['Mystery', 'Supernatural', 'Family'], 'Adults', 'Contemporary', 'Popular']\n}"
    elif n_features == 10:
        interaction_history = "{\"Patriot Games\": [['Thriller', 'Fiction'], ['Adventure', 'War', 'Politics'], 'Adults', 'Contemporary', 'Popular', 'English', 'Paperback', 'Novel', '1951-2000', 'American'], \"The Pillars of the Earth\": [['Historical Fiction'], ['Adventure', 'Love', 'War'], 'Adults', 'Historical', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'British'], \"The Firm\": [['Thriller', 'Mystery'], ['Crime', 'Justice'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American'], \"Insomnia\": [['Horror', 'Fantasy'], ['Psychological', 'Supernatural', 'Mystery'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American'], \"Second Child\": [['Thriller', 'Horror'], ['Mystery', 'Family', 'Crime'], 'Adults', 'Contemporary', 'Popular', 'English', 'Paperback', 'Novel', '1951-2000', 'American'], \"Skeleton Crew\": [['Fiction', 'Horror', 'Thriller'], ['Adventure', 'Survival', 'Courage', 'Betrayal'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American'], \"The Stand: The Complete &amp; Uncut Edition\": [['Science Fiction', 'Horror', 'Fantasy'], ['Survival', 'Good vs Evil', 'Post-Apocalyptic'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American'], \"The Runaway Jury\": [['Thriller', 'Mystery'], ['Crime', 'Justice'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Hardcover', 'Novel', '1951-2000', 'American'], \"The Client\": [['Thriller', 'Mystery'], ['Crime', 'Justice'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American'], \"The Pelican Brief\": [['Thriller', 'Mystery'], ['Crime', 'Justice'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Hardcover', 'Novel', '1951-2000', 'American']}"
        if k == 1:
            example_output = "{\n  \"Nathaniel\": [['Horror', 'Thriller'], ['Mystery', 'Supernatural', 'Family'], 'Adults', 'Contemporary', 'Popular', 'English', 'Paperback', 'Novel', '1951-2000', 'American']\n}"
    elif n_features == 15:
        interaction_history = "{\"Patriot Games\": [['Thriller', 'Fiction'], ['Adventure', 'War', 'Politics'], 'Adults', 'Contemporary', 'Popular', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', 'More than 500', 'Intermediate', 'None', 'Unknown'], \"The Pillars of the Earth\": [['Historical Fiction'], ['Adventure', 'Love', 'War', 'Politics', 'History'], 'Adults', 'Historical', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'British', 'Male', 'More than 500', 'Intermediate', 'None', 'Unknown'], \"The Firm\": [['Thriller', 'Mystery'], ['Crime', 'Justice', 'Betrayal'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', '401-500', 'Intermediate', 'None', 'Unknown'], \"Insomnia\": [['Fiction', 'Horror', 'Fantasy'], ['Psychology', 'Supernatural', 'Identity'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', 'More than 500', 'Intermediate', 'None', 'Unknown'], \"Second Child\": [['Thriller', 'Horror'], ['Mystery', 'Family', 'Betrayal'], 'Adults', 'Contemporary', 'Average', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', '301-400', 'Intermediate', 'None', 'Unknown'], \"Skeleton Crew\": [['Fiction', 'Horror', 'Thriller'], ['Mystery', 'Crime', 'Survival', 'Courage'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', 'More than 500', 'Intermediate', 'None', 'Unknown'], \"The Stand: The Complete &amp; Uncut Edition\": [['Science Fiction', 'Horror', 'Fantasy'], ['Survival', 'Good vs Evil', 'Post-Apocalyptic'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', 'More than 500', 'Intermediate', 'None', 'Unknown'], \"The Runaway Jury\": [['Thriller', 'Mystery'], ['Justice', 'Crime'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Hardcover', 'Novel', '1951-2000', 'American', 'Male', '401-500', 'Intermediate', 'None', 'Unknown'], \"The Client\": [['Thriller', 'Mystery'], ['Crime', 'Justice'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', '401-500', 'Intermediate', 'None', 'Unknown'], \"The Pelican Brief\": [['Thriller', 'Mystery'], ['Crime', 'Justice'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Hardcover', 'Novel', '1951-2000', 'American', 'Male', '301-400', 'Intermediate', 'None', 'Unknown']}"
        if k == 1:
            example_output = "{\n  \"Nathaniel\": [['Horror', 'Thriller'], ['Mystery', 'Supernatural', 'Family'], 'Adults', 'Contemporary', 'Popular', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', '301-400', 'Intermediate', 'None', 'Unknown']\n}"
    return interaction_history, example_output


def generate_prompt_samples_nametax_tax(n_features, k):
    if n_features == 5:
        interaction_history = "{\"Patriot Games\": [['Thriller', 'Fiction'], ['Adventure', 'War', 'Politics', 'Crime'], 'Adults', 'Contemporary', 'Bestseller'], \"The Pillars of the Earth\": [['Historical Fiction'], ['Adventure', 'Love', 'War', 'Politics', 'History'], 'Adults', 'Historical', 'Bestseller'], \"The Firm\": [['Thriller', 'Mystery'], ['Crime', 'Justice', 'Betrayal'], 'Adults', 'Contemporary', 'Bestseller'], \"Insomnia\": [['Horror', 'Fantasy'], ['Psychological', 'Supernatural', 'Mystery'], 'Adults', 'Contemporary', 'Bestseller'], \"Second Child\": [['Thriller', 'Horror'], ['Crime', 'Family', 'Mystery'], 'Adults', 'Contemporary', 'Popular'], \"Skeleton Crew\": [['Fiction', 'Horror', 'Thriller'], ['Survival', 'Horror', 'Mystery'], 'Adults', 'Contemporary', 'Bestseller'], \"The Stand: The Complete &amp; Uncut Edition\": [['Science Fiction', 'Horror', 'Fantasy'], ['Survival', 'Good vs. Evil', 'Apocalyptic'], 'Adults', 'Contemporary', 'Bestseller'], \"The Runaway Jury\": [['Thriller', 'Mystery'], ['Crime', 'Justice', 'Betrayal'], 'Adults', 'Contemporary', 'Bestseller'], \"The Client\": [['Thriller', 'Mystery'], ['Crime', 'Justice', 'Survival'], 'Adults', 'Contemporary', 'Bestseller'], \"The Pelican Brief\": [['Thriller', 'Mystery'], ['Crime', 'Justice', 'Politics'], 'Adults', 'Contemporary', 'Bestseller']}"
        if k == 1:
            example_output = "[['Horror', 'Thriller'], ['Mystery', 'Supernatural', 'Family'], 'Adults', 'Contemporary', 'Popular']"
    elif n_features == 10:
        interaction_history = "{\"Patriot Games\": [['Thriller', 'Fiction'], ['Adventure', 'War', 'Politics'], 'Adults', 'Contemporary', 'Popular', 'English', 'Paperback', 'Novel', '1951-2000', 'American'], \"The Pillars of the Earth\": [['Historical Fiction'], ['Adventure', 'Love', 'War'], 'Adults', 'Historical', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'British'], \"The Firm\": [['Thriller', 'Mystery'], ['Crime', 'Justice'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American'], \"Insomnia\": [['Horror', 'Fantasy'], ['Psychological', 'Supernatural', 'Mystery'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American'], \"Second Child\": [['Thriller', 'Horror'], ['Mystery', 'Family', 'Crime'], 'Adults', 'Contemporary', 'Popular', 'English', 'Paperback', 'Novel', '1951-2000', 'American'], \"Skeleton Crew\": [['Fiction', 'Horror', 'Thriller'], ['Adventure', 'Survival', 'Courage', 'Betrayal'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American'], \"The Stand: The Complete &amp; Uncut Edition\": [['Science Fiction', 'Horror', 'Fantasy'], ['Survival', 'Good vs Evil', 'Post-Apocalyptic'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American'], \"The Runaway Jury\": [['Thriller', 'Mystery'], ['Crime', 'Justice'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Hardcover', 'Novel', '1951-2000', 'American'], \"The Client\": [['Thriller', 'Mystery'], ['Crime', 'Justice'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American'], \"The Pelican Brief\": [['Thriller', 'Mystery'], ['Crime', 'Justice'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Hardcover', 'Novel', '1951-2000', 'American']}"
        if k == 1:
            example_output = "[['Horror', 'Thriller'], ['Mystery', 'Supernatural', 'Family'], 'Adults', 'Contemporary', 'Popular', 'English', 'Paperback', 'Novel', '1951-2000', 'American']"
    elif n_features == 15:
        interaction_history = "{\"Patriot Games\": [['Thriller', 'Fiction'], ['Adventure', 'War', 'Politics'], 'Adults', 'Contemporary', 'Popular', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', 'More than 500', 'Intermediate', 'None', 'Unknown'], \"The Pillars of the Earth\": [['Historical Fiction'], ['Adventure', 'Love', 'War', 'Politics', 'History'], 'Adults', 'Historical', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'British', 'Male', 'More than 500', 'Intermediate', 'None', 'Unknown'], \"The Firm\": [['Thriller', 'Mystery'], ['Crime', 'Justice', 'Betrayal'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', '401-500', 'Intermediate', 'None', 'Unknown'], \"Insomnia\": [['Fiction', 'Horror', 'Fantasy'], ['Psychology', 'Supernatural', 'Identity'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', 'More than 500', 'Intermediate', 'None', 'Unknown'], \"Second Child\": [['Thriller', 'Horror'], ['Mystery', 'Family', 'Betrayal'], 'Adults', 'Contemporary', 'Average', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', '301-400', 'Intermediate', 'None', 'Unknown'], \"Skeleton Crew\": [['Fiction', 'Horror', 'Thriller'], ['Mystery', 'Crime', 'Survival', 'Courage'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', 'More than 500', 'Intermediate', 'None', 'Unknown'], \"The Stand: The Complete &amp; Uncut Edition\": [['Science Fiction', 'Horror', 'Fantasy'], ['Survival', 'Good vs Evil', 'Post-Apocalyptic'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', 'More than 500', 'Intermediate', 'None', 'Unknown'], \"The Runaway Jury\": [['Thriller', 'Mystery'], ['Justice', 'Crime'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Hardcover', 'Novel', '1951-2000', 'American', 'Male', '401-500', 'Intermediate', 'None', 'Unknown'], \"The Client\": [['Thriller', 'Mystery'], ['Crime', 'Justice'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', '401-500', 'Intermediate', 'None', 'Unknown'], \"The Pelican Brief\": [['Thriller', 'Mystery'], ['Crime', 'Justice'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Hardcover', 'Novel', '1951-2000', 'American', 'Male', '301-400', 'Intermediate', 'None', 'Unknown']}"
        if k == 1:
            example_output = "[['Horror', 'Thriller'], ['Mystery', 'Supernatural', 'Family'], 'Adults', 'Contemporary', 'Popular', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', '301-400', 'Intermediate', 'None', 'Unknown']"
    return interaction_history, example_output


def generate_prompt_samples_tax_tax(n_features, k):
    if n_features == 5:
        interaction_history = "[[['Thriller', 'Fiction'], ['Adventure', 'War', 'Politics', 'Crime'], 'Adults', 'Contemporary', 'Bestseller'], [['Historical Fiction'], ['Adventure', 'Love', 'War', 'Politics', 'History'], 'Adults', 'Historical', 'Bestseller'], [['Thriller', 'Mystery'], ['Crime', 'Justice', 'Betrayal'], 'Adults', 'Contemporary', 'Bestseller'], [['Horror', 'Fantasy'], ['Psychological', 'Supernatural', 'Mystery'], 'Adults', 'Contemporary', 'Bestseller'], [['Thriller', 'Horror'], ['Crime', 'Family', 'Mystery'], 'Adults', 'Contemporary', 'Popular'], [['Fiction', 'Horror', 'Thriller'], ['Survival', 'Horror', 'Mystery'], 'Adults', 'Contemporary', 'Bestseller'], [['Science Fiction', 'Horror', 'Fantasy'], ['Survival', 'Good vs. Evil', 'Apocalyptic'], 'Adults', 'Contemporary', 'Bestseller'], [['Thriller', 'Mystery'], ['Crime', 'Justice', 'Betrayal'], 'Adults', 'Contemporary', 'Bestseller'], [['Thriller', 'Mystery'], ['Crime', 'Justice', 'Survival'], 'Adults', 'Contemporary', 'Bestseller'], [['Thriller', 'Mystery'], ['Crime', 'Justice', 'Politics'], 'Adults', 'Contemporary', 'Bestseller']]"
        if k == 1:
            example_output = "[['Horror', 'Thriller'], ['Mystery', 'Supernatural', 'Family'], 'Adults', 'Contemporary', 'Popular']"
    elif n_features == 10:
        interaction_history = "[[['Thriller', 'Fiction'], ['Adventure', 'War', 'Politics'], 'Adults', 'Contemporary', 'Popular', 'English', 'Paperback', 'Novel', '1951-2000', 'American'], [['Historical Fiction'], ['Adventure', 'Love', 'War'], 'Adults', 'Historical', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'British'], [['Thriller', 'Mystery'], ['Crime', 'Justice'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American'], [['Horror', 'Fantasy'], ['Psychological', 'Supernatural', 'Mystery'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American'], [['Thriller', 'Horror'], ['Mystery', 'Family', 'Crime'], 'Adults', 'Contemporary', 'Popular', 'English', 'Paperback', 'Novel', '1951-2000', 'American'], [['Fiction', 'Horror', 'Thriller'], ['Adventure', 'Survival', 'Courage', 'Betrayal'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American'], [['Science Fiction', 'Horror', 'Fantasy'], ['Survival', 'Good vs Evil', 'Post-Apocalyptic'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American'], [['Thriller', 'Mystery'], ['Crime', 'Justice'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Hardcover', 'Novel', '1951-2000', 'American'], [['Thriller', 'Mystery'], ['Crime', 'Justice'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American'], [['Thriller', 'Mystery'], ['Crime', 'Justice'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Hardcover', 'Novel', '1951-2000', 'American']]"
        if k == 1:
            example_output = "[['Horror', 'Thriller'], ['Mystery', 'Supernatural', 'Family'], 'Adults', 'Contemporary', 'Popular', 'English', 'Paperback', 'Novel', '1951-2000', 'American']"
    elif n_features == 15:
        interaction_history = "[[['Thriller', 'Fiction'], ['Adventure', 'War', 'Politics'], 'Adults', 'Contemporary', 'Popular', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', 'More than 500', 'Intermediate', 'None', 'Unknown'], [['Historical Fiction'], ['Adventure', 'Love', 'War', 'Politics', 'History'], 'Adults', 'Historical', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'British', 'Male', 'More than 500', 'Intermediate', 'None', 'Unknown'], [['Thriller', 'Mystery'], ['Crime', 'Justice', 'Betrayal'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', '401-500', 'Intermediate', 'None', 'Unknown'], [['Fiction', 'Horror', 'Fantasy'], ['Psychology', 'Supernatural', 'Identity'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', 'More than 500', 'Intermediate', 'None', 'Unknown'], [['Thriller', 'Horror'], ['Mystery', 'Family', 'Betrayal'], 'Adults', 'Contemporary', 'Average', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', '301-400', 'Intermediate', 'None', 'Unknown'], [['Fiction', 'Horror', 'Thriller'], ['Mystery', 'Crime', 'Survival', 'Courage'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', 'More than 500', 'Intermediate', 'None', 'Unknown'], [['Science Fiction', 'Horror', 'Fantasy'], ['Survival', 'Good vs Evil', 'Post-Apocalyptic'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', 'More than 500', 'Intermediate', 'None', 'Unknown'], [['Thriller', 'Mystery'], ['Justice', 'Crime'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Hardcover', 'Novel', '1951-2000', 'American', 'Male', '401-500', 'Intermediate', 'None', 'Unknown'], [['Thriller', 'Mystery'], ['Crime', 'Justice'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', '401-500', 'Intermediate', 'None', 'Unknown'], [['Thriller', 'Mystery'], ['Crime', 'Justice'], 'Adults', 'Contemporary', 'Bestseller', 'English', 'Hardcover', 'Novel', '1951-2000', 'American', 'Male', '301-400', 'Intermediate', 'None', 'Unknown']]"
        if k == 1:
            example_output = "[['Horror', 'Thriller'], ['Mystery', 'Supernatural', 'Family'], 'Adults', 'Contemporary', 'Popular', 'English', 'Paperback', 'Novel', '1951-2000', 'American', 'Male', '301-400', 'Intermediate', 'None', 'Unknown']"
    return interaction_history, example_output


if n_features == 5:
    taxonomy = {
        # top-5
        "Genre": ["Fiction", "Non-Fiction", "Science Fiction", "Fantasy", "Mystery", "Thriller", "Romance", "Historical Fiction", "Biography", "Self-Help", "Children's", "Young Adult", "Humor", "Classics", "Poetry", "Graphic Novel", "Satire", "Urban Fantasy", "Dystopian"],
        "Theme": ["Adventure", "Friendship", "Love", "War", "Crime", "Fantasy", "Science", "Religion", "Philosophy", "Politics", "History", "Technology", "Magic", "Travel", "Art", "Music", "Sports", "Family", "Identity", "Growth", "Ethics", "Nature", "Survival", "Courage", "Betrayal", "Justice", "Freedom"],
        "Audience Age Group": ["Children", "Teens", "Adults", "Seniors", "All Ages"],
        "Setting": ["Contemporary", "Historical", "Future", "Alternate Reality", "Urban", "Rural", "Space", "Fantasy World"],
        "Popularity": ["Bestseller", "Popular", "Average", "Niche"]
    }
    if mode == 'nametax_nametax':
        interaction_history, sample_recommendations = generate_prompt_samples_nametax_nametax(n_features, k)
    elif mode == 'nametax_tax':
        interaction_history, sample_recommendations = generate_prompt_samples_nametax_tax(n_features, k)
    elif mode == 'tax_tax':
        interaction_history, sample_recommendations = generate_prompt_samples_tax_tax(n_features, k)
    # interaction_history, sample_recommendations = generate_prompt_samples_nametax_tax(n_features, k)
    # interaction_history, sample_recommendations = generate_prompt_samples_tax_tax(n_features, k)

    # name + taxonomy prompt
    prompt_nametax = f"""
    <Instruction>
    You are a book recommender system. Given a list of books the user has read before, please recommend {k} book(s) in a list of features following the format of the given taxonomy.
    book taxonomy: {taxonomy}
    The input and output format of each book is represented by a list of features. 
    Given the book taxonomy and a list of books the user has read before, please recommend {k} new type(s) of book(s) that align(s) with the user's preferences.
    <End of Instruction>

    <Restrictions>
    Output the recommended book with their type in a list of features:
    [[Genre], [Theme], [Audience Age Group], [Setting], [Popularity]]
    Fill each feature in '[]' with the corresponding value(s) based on the book taxonomy. The output only consists of the book title and a list of features without any other texts.",
    <End of Restrictions>

    <Example>
    Input:
    The user has read with the following books before:{interaction_history}\n
    
    Output:
    {sample_recommendations}
    <End of Example>

    Input:
    """

    # taxonomy prompt
    prompt_tax = f"""
    <Instruction>
    You are a book recommender system. Given a list of books the user has read before, please recommend {k} book(s) in a list of features following the format of the given taxonomy.
    book taxonomy: {taxonomy}
    The input and output format of each book is represented by a list of features. 
    Given the book taxonomy and a list of books the user has read before, please recommend {k} new type(s) of book(s) that align(s) with the user's preferences.
    <End of Instruction>

    <Restrictions>
    Output the recommended type(s) of book(s) in a list of lists as the format of the taxonomy:
    [[Genre], [Theme], [Audience Age Group], [Setting], [Popularity]]
    Fill each feature in '[]' with the corresponding value(s) based on the book taxonomy. The output only consists of a list of lists without any other texts.",
    <End of Restrictions>

    <Example>
    Input:
    The user has read with the following books before:{interaction_history}\n
    
    Output:
    {sample_recommendations}
    <End of Example>

    Input:
    """


elif n_features == 10:
    taxonomy = {
        # top-5
        "Genre": ["Fiction", "Non-Fiction", "Science Fiction", "Fantasy", "Mystery", "Thriller", "Romance", "Historical Fiction", "Biography", "Self-Help", "Children's", "Young Adult", "Humor", "Classics", "Poetry", "Graphic Novel", "Satire", "Urban Fantasy", "Dystopian"],
        "Theme": ["Adventure", "Friendship", "Love", "War", "Crime", "Fantasy", "Science", "Religion", "Philosophy", "Politics", "History", "Technology", "Magic", "Travel", "Art", "Music", "Sports", "Family", "Identity", "Growth", "Ethics", "Nature", "Survival", "Courage", "Betrayal", "Justice", "Freedom"],
        "Audience Age Group": ["Children", "Teens", "Adults", "Seniors", "All Ages"],
        "Setting": ["Contemporary", "Historical", "Future", "Alternate Reality", "Urban", "Rural", "Space", "Fantasy World"],
        "Popularity": ["Bestseller", "Popular", "Average", "Niche"],

        # top-10
        "Language": ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Italian", "Portuguese", "Russian", "Other"],
        "Format": ["Hardcover", "Paperback", "Ebook", "Audiobook"],
        "Book Length": ["Short Story", "Novella", "Novel", "Series"],
        "Publication Year": ["Pre-1900", "1900-1950", "1951-2000", "2001-2010", "2011-2020", "2021-Present"],
        "Author Nationality": ["American", "British", "Canadian", "Australian", "Indian", "Chinese", "Japanese", "French", "German", "Russian", "Italian", "Spanish", "Brazilian", "Mexican", "South African", "Haitian", "Other"]
    }
    if mode == 'nametax_nametax':
        interaction_history, sample_recommendations = generate_prompt_samples_nametax_nametax(n_features, k)
    elif mode == 'nametax_tax':
        interaction_history, sample_recommendations = generate_prompt_samples_nametax_tax(n_features, k)
    elif mode == 'tax_tax':
        interaction_history, sample_recommendations = generate_prompt_samples_tax_tax(n_features, k)
    # interaction_history, sample_recommendations = generate_prompt_samples_nametax_tax(n_features, k)
    # interaction_history, sample_recommendations = generate_prompt_samples_tax_tax(n_features, k)


    prompt_nametax = f"""
    <Instruction>
    You are a book recommender system. Given a list of books the user has read before, please recommend {k} book(s) in a list of features following the format of the given taxonomy.
    book taxonomy: {taxonomy}
    The input and output format of each book is represented by a list of features. 
    Given the book taxonomy and a list of books the user has read before, please recommend {k} new type(s) of book(s) that align(s) with the user's preferences.
    <End of Instruction>

    <Restrictions>
    Output the recommended book with their type in a list of features:
    [[Genre], [Theme], [Audience Age Group], [Setting], [Popularity], [Language], [Format], [Book Length], [Publication Year], [Author National]]
    Fill each feature in '[]' with the corresponding value(s) based on the book taxonomy. The output only consists of the book title and a list of features without any other texts.",
    <End of Restrictions>

    <Example>
    Input:
    The user has read with the following books before:{interaction_history}\n 
    Output:
    {sample_recommendations}
    <End of Example>

    Input:
    """

    # taxonomy prompt
    prompt_tax = f"""
    <Instruction>
    You are a book recommender system. Given a list of books the user has read before, please recommend {k} book(s) in a list of features following the format of the given taxonomy.
    book taxonomy: {taxonomy}
    The input and output format of each book is represented by a list of features. 
    Given the book taxonomy and a list of books the user has read before, please recommend {k} new type(s) of book(s) that align(s) with the user's preferences.
    <End of Instruction>

    <Restrictions>
    Output the recommended type(s) of book(s) in a list of lists as the format of the taxonomy:
    [[Genre], [Theme], [Audience Age Group], [Setting], [Popularity], [Language], [Format], [Book Length], [Publication Year], [Author National]]
    Fill each feature in '[]' with the corresponding value(s) based on the book taxonomy. The output only consists of a list of lists without any other texts.",
    <End of Restrictions>

    <Example>
    Input:
    The user has read with the following books before:{interaction_history}\n
    
    Output:
    {sample_recommendations}
    <End of Example>

    Input:
    """





elif n_features == 15:
    taxonomy = {
        # top-5
        "Genre": ["Fiction", "Non-Fiction", "Science Fiction", "Fantasy", "Mystery", "Thriller", "Romance", "Historical Fiction", "Biography", "Self-Help", "Children's", "Young Adult", "Humor", "Classics", "Poetry", "Graphic Novel", "Satire", "Urban Fantasy", "Dystopian"],
        "Theme": ["Adventure", "Friendship", "Love", "War", "Crime", "Fantasy", "Science", "Religion", "Philosophy", "Politics", "History", "Technology", "Magic", "Travel", "Art", "Music", "Sports", "Family", "Identity", "Growth", "Ethics", "Nature", "Survival", "Courage", "Betrayal", "Justice", "Freedom"],
        "Audience Age Group": ["Children", "Teens", "Adults", "Seniors", "All Ages"],
        "Setting": ["Contemporary", "Historical", "Future", "Alternate Reality", "Urban", "Rural", "Space", "Fantasy World"],
        "Popularity": ["Bestseller", "Popular", "Average", "Niche"],

        # top-10
        "Language": ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Italian", "Portuguese", "Russian", "Other"],
        "Format": ["Hardcover", "Paperback", "Ebook", "Audiobook"],
        "Book Length": ["Short Story", "Novella", "Novel", "Series"],
        "Publication Year": ["Pre-1900", "1900-1950", "1951-2000", "2001-2010", "2011-2020", "2021-Present"],
        "Author Nationality": ["American", "British", "Canadian", "Australian", "Indian", "Chinese", "Japanese", "French", "German", "Russian", "Italian", "Spanish", "Brazilian", "Mexican", "South African", "Haitian", "Other"],
        
        # top-15
        "Author Gender": ["Male", "Female", "Non-Binary", "Other", "Unknown"],
        "Page Count": ["Less than 100", "100-200", "201-300", "301-400", "401-500", "More than 500"],
        "Reading Level": ["Beginner", "Intermediate", "Advanced"],
        "Award": ["Pulitzer Prize", "Man Booker Prize", "National Book Award", "Hugo Award", "Nebula Award", "None"],
        "Rating": ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars", "Unknown"]
    }
    if mode == 'nametax_nametax':
        interaction_history, sample_recommendations = generate_prompt_samples_nametax_nametax(n_features, k)
    elif mode == 'nametax_tax':
        interaction_history, sample_recommendations = generate_prompt_samples_nametax_tax(n_features, k)
    elif mode == 'tax_tax':
        interaction_history, sample_recommendations = generate_prompt_samples_tax_tax(n_features, k)
    # interaction_history, sample_recommendations = generate_prompt_samples_nametax_tax(n_features, k)
    # interaction_history, sample_recommendations = generate_prompt_samples_tax_tax(n_features, k)


    prompt_nametax = f"""
    <Instruction>
    You are a book recommender system. Given a list of books the user has read before, please recommend {k} book(s) in a list of features following the format of the given taxonomy.
    book taxonomy: {taxonomy}
    The input and output format of each book is represented by a list of features. 
    Given the book taxonomy and a list of books the user has read before, please recommend {k} new type(s) of book(s) that align(s) with the user's preferences.
    <End of Instruction>

    <Restrictions>
    Output the recommended book with their type in a list of features:
    [[Genre], [Theme], [Audience Age Group], [Setting], [Popularity], [Language], [Format], [Book Length], [Publication Year], [Author National], [Author Gender], [Page Count], [Reading Level], [Award], [Rating]]
    Fill each feature in '[]' with the corresponding value(s) based on the book taxonomy. The output only consists of the book title and a list of features without any other texts.",
    <End of Restrictions>

    <Example>
    Input:
    The user has read with the following books before:{interaction_history}\n 
    Output:
    {sample_recommendations}
    <End of Example>

    Input:
    """

    # taxonomy prompt
    prompt_tax = f"""
    <Instruction>
    You are a book recommender system. Given a list of books the user has read before, please recommend {k} book(s) in a list of features following the format of the given taxonomy.
    book taxonomy: {taxonomy}
    The input and output format of each book is represented by a list of features. 
    Given the book taxonomy and a list of books the user has read before, please recommend {k} new type(s) of book(s) that align(s) with the user's preferences.
    <End of Instruction>

    <Restrictions>
    Output the recommended type(s) of book(s) in a list of lists as the format of the taxonomy:
    [[Genre], [Theme], [Audience Age Group], [Setting], [Popularity], [Language], [Format], [Book Length], [Publication Year], [Author National], [Author Gender], [Page Count], [Reading Level], [Award], [Rating]]
    Fill each feature in '[]' with the corresponding value(s) based on the book taxonomy. The output only consists of a list of lists without any other texts.",
    <End of Restrictions>

    <Example>
    Input:
    The user has read with the following books before:{interaction_history}\n
    
    Output:
    {sample_recommendations}
    <End of Example>

    Input:
    """

if mode == 'tax_tax':
    prompt = prompt_tax
else:
    prompt = prompt_nametax

# Initialize the OpenAI API client
client = AzureOpenAI(
  api_key = "c6af48fe651d44bb80477d9f17918c3d",
  api_version = '2023-05-15',
  azure_endpoint = "https://gpt-35-1106.openai.azure.com"
)


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

output_dir = 'outputs/'
with open(output_dir + output_path, 'w') as file:
    json.dump(data, file, indent=2)