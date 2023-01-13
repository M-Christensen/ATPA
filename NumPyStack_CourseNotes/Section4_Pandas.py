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
n = 10000
rad1 = np.sqrt(10)
rad2 = np.sqrt(1)
beyond = 5

# x2 = np.random.random(n) * 15 - 7.5
# print(x1)
# print(x1.mean())
# print(x2)


# %%
def create_df(n, radius, radius2, beyond, beyond2):
    outer_circle = np.random.randn(2*n) * (radius*2+beyond) - (radius+beyond/2)
    inner_circle = np.random.randn(2*n) * (radius2*2+beyond2) - (radius2+beyond2/2)
    y = np.concatenate([np.zeros(n), np.ones(n)])
    
    dat = pd.DataFrame(data = {'x1': np.concatenate([outer_circle[:int(n)], inner_circle[:int(n)]]), 
                               'x2': np.concatenate([outer_circle[int(n):], inner_circle[int(n):]])})
    return dat.assign(
                    x1_2 = lambda x: x['x1']**2,
                    x2_2 = lambda x: x['x2']**2,
                    x1_x2 = lambda x: x['x1']*x['x2'],
                    y = y
                    )

df = create_df(n, rad1, rad2, beyond, beyond)

# print(df)
# %%
print(df.groupby("y").mean())
df.groupby("y").agg(['mean', 'std'])
# %%
plt.scatter(np.sign(df['x1'])*df['x1_2'], np.sign(df['x2'])*df['x2_2'], c=df['y'])
# %%
x1 = np.repeat(np.arange(-10, 10.5, 0.5), 2)
x2 = np.repeat(np.concatenate([np.arange(0, 10.5, 0.5), np.arange(0, 10, 0.5)[::-1]]),2) * np.tile(np.array([-1,1]), int(len(x1)/2))

print(x1)
print(x2)
# %%
plt.scatter(x1**2,x2**2)

# %%
plt.scatter(x1+np.random.randn(len(x1))/2, x2+np.random.randn(len(x2))/2)
# %%
# def create_diamond_df(n, radius1, radius2, std1, std2):
#     outer_diamond = 




#%%
# (x1-h)**2 + (x2-k)**2 = r**2
h = 0
k = 0
r = 3
gap = 0.01

# x1 = np.tile(np.arange(-r, r+gap, gap),2)
x1 = np.arange(-r,r+gap,gap)
x2 = np.concatenate([np.sqrt(r**2 - (x1-h)**2), -1*np.sqrt(r**2 - (x1-h)**2)])

# %%
fig = plt.figure()
ax = fig.add_subplot()
ax.set_aspect('equal', adjustable='box')
plt.scatter(np.tile(x1,2) + np.random.rand(len(x2))/4,x2 + np.random.randn(len(x2))/4)
# %%
fig = plt.figure()
ax = fig.add_subplot()
ax.set_aspect('equal', adjustable='box')
plt.scatter(np.tile(x1,2), x2)
# %%
