import numpy as np
import pandas as pd
import json


# define function to reed json
def get_df(fn, limit=None):
    json_lines = []
    line_nr = 1
    with open(fn) as f:
        for line in f:
            if limit and line_nr == limit:
                break
            json_line = json.loads(line)
            json_lines.append(json_line)
            line_nr += 1
    df = pd.DataFrame(json_lines)
    return df

# define a list with all symbols to be removed
punctuation = ['?', '.', ':', ':', '!', '"', '(', ')', '-', '$', ',', '+', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

# function to remove these symbols
def remove_punctuation(text):    
    cleaned_text = "".join(u for u in text if u not in punctuation)
    return cleaned_text