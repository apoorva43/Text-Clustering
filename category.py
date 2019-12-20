''' Code to modifiy category for text clustering '''

import pandas as pd
import numpy as np
from collections import defaultdict


def clean_data():
    data = pd.read_csv('data.csv') # read data
    data = data.rename(columns = {"Please Check": "NEW CATEGORY"})
    # remove null values
    data = data.dropna(subset = ['DESCRIPTION'])
    
    old_category, new_category, comment = [], [], []
    for i in data.index:
        old_category.append(data['CATEGORY'][i])
        new_category.append("")
        comment.append(data['DESCRIPTION'][i])
    
    count_ = defaultdict(int)
    for i in range(len(old_category)):
        string = old_category[i].strip().split(',')
        count_[string[0]] += 1
    
    for i in range(len(old_category)):
        string = old_category[i].strip().split(',')
        # club sparse classes
        if count_[string[0]] < 200 or count_[string[0]] == 346: 
            new_category[i] = 'Others'
        elif count_[string[0]] < 330:
            new_category[i] = 'Ogling/Facial Expressions/Staring'
        elif count_[string[0]] == 585 or count_[string[0]] == 462:
            new_category[i] = 'Rape / Sexual Assault'
        else:
            new_category[i] = string[0]
    
    return new_category, comment



new_category, comment = clean_data()
data_new = {'DESCRIPTION': comment, 'CATEGORY': new_category}
# new dataframe
data_new_df = pd.DataFrame(data_new)
# print(data_new_df.head())
print(data_new_df['CATEGORY'].value_counts())
# write to new csv file
data_new_df.to_csv('category.csv') 
