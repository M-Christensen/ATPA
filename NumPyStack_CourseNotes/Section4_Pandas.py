#### Code for Section 4 - Pandas of the Deep Learning Prerequisities course

' Lesson 21 - Loading in Data'
import pandas as pd
import numpy as np

# import raw csv file
# !wget https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/sbux.csv

df = pd.read_csv('Data/sbux.csv')

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
df2 = pd.read_csv('Data/sbux.csv', index_col = 'date')
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
smalldf.to_csv("Data/output.csv", index=False)

' Lesson 23. The apply() function'
# apply function will perform a function across the rows or columns of a data frame
# for example, extract the last two numbers from the year
def year_split(row):
    return int(row['date'].split('-')[0][2:])

df.apply(year_split, axis=1)


' Lesson 24. Plotting with Pandas'
# pandas automatically create instance methods on columns so we can plot them directly
df['open'].hist();

df['open'].plot();

df[['open', 'high', 'low', 'close']].plot.box();


from pandas.plotting import scatter_matrix
scatter_matrix(df[['open', 'high', 'low', 'close']],
                alpha=0.2, figsize=(6,6))
                
' Lesson 25. Pandas Exercise '
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
n = 10
rad1 = np.sqrt(10)
rad2 = np.sqrt(5)
beyond = 5

# x2 = np.random.random(n) * 15 - 7.5
# print(x1)
# print(x1.mean())
# print(x2)

def create_df(n, radius, radius2, beyond, beyond2):
    outer_circle = np.random.random(2*n) * (radius*2+beyond) - (radius+beyond/2)
    inner_circle = np.random.random(n) * (radius*2+beyond) - (radius+beyond/2)
    y = np.array(np.zeros(n))
    # return {'x1': outer_circle[:int(n/2)], 
    #         'x2': outer_circle[int(n/2):]}
    return pd.DataFrame(data = {'x1': outer_circle[:int(n)], 
                                'x2': outer_circle[int(n):],
                                'y': y})

df = create_df(n, rad1, rad2, beyond, beyond)




# x1_2 = x1**2
# x1_2
# pd.DataFrame()
# plt.scatter(x1_2[:n/2], x1_2[n/2:])



# %%
