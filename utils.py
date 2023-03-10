import os
import pandas as pd
import json

def load_data(dataset='train'):
    """
    Load the dataset from the 'data' folder
    """
    if dataset not in ['train', 'test', 'val']:
        raise ValueError('dataset param must be either train, test or val')
    filename = dataset + ".txt"
    data_path = os.path.join(os.path.dirname(__file__), 'data', filename)
    with open(data_path, 'r') as f:
        data = f.read().splitlines()
    
    df = pd.DataFrame(data, columns=['text'])
    df = df.text.str.split(';', expand=True)
    df.columns = ['text', 'label']

    return df


def transform_dataset(df):
    """
    Transform the dataset by applying the following functions:
    - join_single_letters
    - expand_contractions
    """
    df = join_single_letters(df)
    df = expand_contractions(df)
    return df

def expand_contractions(df):
    """
    Performs the expand_contractions_in_text function on each row of the dataframe.
    """
    contractions = get_contractions_list()
    
    df['text'] = df['text'].apply(lambda x: expand_contractions_in_text(x, contractions))
    return df


def expand_contractions_in_text(text, contractions):
    """
    Expand contractions in the text.
    """
    new_text = text.split(' ')
    new_text = [contractions[word] if word in contractions else word for word in new_text]
    return " ".join(new_text)


def get_contractions_list():
    """
    Load the contractions list from the 'contractions.json' file.
    """
    filename = 'contractions.json'
    data_path = os.path.join(os.path.dirname(__file__), filename)
    with open(data_path, 'r') as f:
        contractions = json.load(f)
    
    transformed_contractions = transform_contractions_list(contractions)

    return transformed_contractions


def transform_contractions_list(contractions):
    """
    Transform the contractions list to a dictionary with the keys in lowercase and without apostrophes.
    """
    transformed_contractions = {}
    for key, value in contractions.items():
        new_key = key.replace("'", "").lower()
        transformed_contractions[new_key] = value.lower()
    return transformed_contractions


def join_single_letters(df):
    """
    Performs the join_single_letters_in_text function on each row of the dataframe.
    """
    df['text'] = df['text'].apply(lambda x: join_single_letters_in_text(x))
    return df


def join_single_letters_in_text(text):
    """
    Join single letters to the previous word. It aims to avoid splitting words like "I'm" or "I'll" into "I" and "m" or "I" and "ll".
    """
    splitted_text = text.split(' ')
    new_text = []

    for i, word in enumerate(splitted_text):
        if len(word) == 1 and word != 'i' and word != 'a':
            new_text[-1] = new_text[-1] + word
        else:
            new_text.append(word)

    return " ".join(new_text)
