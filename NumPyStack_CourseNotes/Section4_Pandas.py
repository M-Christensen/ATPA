#### Code for Section 4 - Pandas of the Deep Learning Prerequisities course

' Lesson 21 - Loading in Data'
import pandas as pd

# import raw csv file
# !wget https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/sbux.csv

df = pd.read_csv('sbux.csv')

# simple views of the data
type(df)
df.head()
df.info()

' Lesson 22. Selecting Rows and Columns '
# to index into a pandas data frame, we use the loc and iloc statements
    # loc is used for non-integer values (columns names)
    # iloc is for integers

# selecting columns
df.columns
df[['open', 'close']]

type(df['open']) # type is a series
type(df[['open', 'close']]) # type is a pandas dataframe

# selecting rows
df.iloc[0] # iloc must be integer row values
df.loc[0] # this would not work if our data frame did not have integer indices - dates, for example


# read data in with new index
df2 = pd.read_csv('sbux.csv', index_col = 'date')
df2.head()

df2.loc[0]
df2.loc['2013-02-08']
df2.iloc[0]

# subsetting the pd dataframe
df[df['open'] > 64]
df[df['Name'] != 'SBUX'] # all Name are SBUX

type(df['Name'] != 'SBUX') # individual rows, columns, and vectors of booleans are Series

# converting pd data frames to np arrays
A = df.values
A    # since there are strings in the df, this array is an "object" type

B = df[['open', 'close']].values
B

# output to a csv
smalldf = df[['open']]
smalldf.to_csv("output.csv", index=False)