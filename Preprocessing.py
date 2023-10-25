import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings('ignore')

# ignore warnings
def pre_process(file, rows):
    data = pd.read_csv(file, nrows=rows)
    data = replace_quotes(data=data)
    data = to_lowercase(data=data)

    data = normalize_cols(data=data)
    data = encode(data=data)
    data.to_csv('./dating.csv', index=False)

    test_data = data.sample(frac=0.2, random_state=25)
    test_data.to_csv('./testSet.csv', index=False)
    training_data = data.drop(test_data.index)
    training_data.to_csv('./trainingSet.csv', index=False)


# function that strips single quotes (') of three dimensions of the data set.
# also counts and outputs the number of datapoints that have quotes replaced
def replace_quotes(data):
    replace_cnt = 0
    for i in range(len(data.loc[:, 'race'])):
        if '\'' in data.loc[:, 'race'][i]:
            replace_cnt += 1
        if '\'' in data.loc[:, 'race_o'][i]:
            replace_cnt += 1
        if '\'' in data.loc[:, 'field'][i]:
            replace_cnt += 1

    data['race'] = data['race'].str.replace('\'', '')
    data['race_o'] = data['race_o'].str.replace('\'', '')
    data['field'] = data['field'].str.replace('\'', '')
    return data


# function that sets all characters in field dimension to lowercase
# also counts and outputs the number of datapoints set to lowercase
def to_lowercase(data):
    lower_data = data['field'].str.lower()
    lower_cnt = 0
    for (lower, row) in zip(lower_data, data['field']):
        if lower != row:
            lower_cnt += 1
    data['field'] = lower_data

    return data

# function that normalizes the values in the following columns with each other:
# [attractive_important, sincere_important, intelligence_important, funny_important,
#  ambition_important, shared_interests_important]
# Function also normalizes the values in the following columns with each other:
# ["pref_o_attractive", "pref_o_sincere", "pref_o_intelligence", "pref_o_funny",
#  "pref_o_ambitious", "pref_o_shared_interests"]
def normalize_cols(data):

    participant_cols = ["attractive_important", "sincere_important", "intelligence_important",
            "funny_important", "ambition_important", "shared_interests_important"]

    partner_cols = ["pref_o_attractive", "pref_o_sincere", "pref_o_intelligence", "pref_o_funny",
            "pref_o_ambitious", "pref_o_shared_interests"]
    i = 0
    for index, row in data.iterrows():
        participant_total = 0
        partner_total = 0
        for col in participant_cols:
            participant_total += row[col]
        for col in partner_cols:
            partner_total += row[col]

        for col in participant_cols:
            data.at[i, col] = row[col] / participant_total
        for col in partner_cols:
            data.at[i, col] = row[col] / partner_total
        i += 1
    return data


def encode(data):
    # start with race for now

    #data.to_csv('./after_sorting.csv',index=False)

    cols = ['gender', 'race', 'race_o', 'field']
    attrs = ['female', 'Black/African American', 'Other', 'economics']
    print_cols = {}

    for col in cols:
        data.sort_values(by=[col], inplace=True)
        unique = data[col].unique()
        col_loc = data.columns.get_loc(col)
        initial_col = col_loc
        for attr in range(len(unique) - 1):
            new_col = np.zeros(shape=(len(data), 1))
            data.insert(loc=col_loc + 1, column='{0}_{1}'.format(col, unique[attr]), value=new_col, allow_duplicates=True)
            col_loc += 1
        print_cols[col] = (initial_col, col_loc)

        for row in range(len(data)):
            val = data.loc[row, col]
            if val != unique[-1]:
                data.loc[row, '{0}_{1}'.format(col, val)] = 1

        data.drop(columns=[col], axis=1, inplace=True)
    columns = data.columns
    j = 0
    for col in print_cols.keys():
        start, end = print_cols[col]
        print("Mapped vector for {0} in column {1}: ".format(cols[j], attrs[j]), end="")
        print('[', end="")
        for i in range(start, end):
            if columns[i] == 'gender_female' or columns[i] == 'race_Black/African American' \
                    or columns[i] == 'race_o_Other' or columns[i] == 'field_economics':
                print(' 1 ', end="")
            else:
                print(' 0 ', end="")
        print(']')
        j += 1




    return data


if __name__ == '__main__':
    pre_process('./dating-full.csv', rows=6500)