# Dataframes
import pandas as pd
import numpy as np

# Language detection
import fasttext as ft
from pycountry import pycountry

def language_processing(df_lang, verbose=False, cleaned=True, accuracy=0.95):

    # Load pretrained model
    fasttext_model = ft.load_model('../modeling/lid.176.bin')

    # Initiate empty language list
    language_list = []

    for row in df_lang['text']:
        row = row.replace("\n"," ")                                     # replace \n with " "
        label = fasttext_model.predict(row, k=-1, threshold=accuracy)   # predict language per row with a certainty of at least 95%
        language_list.append(label)                                     # append result to list

    # Set language list as new column in dataframe
    language_df = pd.DataFrame(language_list, columns=['language', 'probability'])
    df_lang['language'] = language_df['language'].astype(str)
    df_lang['language'] = df_lang['language'].str.replace('label',"").str.replace(r"[^a-zA-Z ]+","").str.strip();

    # Convert iso639-1 codes to language names
    names = []
    for i in df_lang['language']:
        try:
            language_names = pycountry.languages.get(alpha_2=i).name
        except:
            language_names = None
        names.append(language_names)

    # Get final language column in dataframe
    names_df = pd.DataFrame(names, columns=['names'])
    df_lang['language'] = names_df['names']

    # Generating the Results for the output
    uniq = df_lang['language']                      # Put the languages in a list
    uniq = list(filter((None).__ne__,uniq));        # Get rid of the None values
    uniq = set(uniq)                                # Convert to set to have the unique values

    en = names.count('English')                     # Count no. of 'English' in names

    most = max(set(names), key=names.count)         # Get the most occurring language in names

    whole = df_lang.shape[0]                        # Whole number of entries 
    nones = sum(x is None for x in names)           # Number of nones
    fraction = round(((whole - nones)*100)/whole,2) # Percentage of classified languages

    if cleaned:
        df_lang = df_lang[df_lang['language'] == 'English']

    if verbose:
        # Output
        print(f"In our Dataset we have a total of {df_lang['language'].nunique()} different languages,")
        print(f"classified with an accuracy of at least {accuracy}.")
        print('---'*50)
        print(f"These languages are (unsorted): \n{uniq}")
        print('---'*50)
        print(f"The classified languages represent {fraction} % of the dataset")
        print('---'*50)
        print(f"The most occurring language is {most}, it occurred {en} times")
        print('---'*50)
        print(f"The algorithm was not able to classify {nones} reviews.")

    return df_lang