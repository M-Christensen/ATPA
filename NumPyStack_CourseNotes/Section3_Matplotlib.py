#### Code for Section 3 - Matplotlib of the Deep Learning Prerequisities course
# Plotting in VSCode .py files: https://donjayamanne.github.io/pythonVSCodeDocs/docs/jupyter_examples/

' Lesson 14. Line Chart '
#%%
import numpy as np
import matplotlib.pyplot as plt

#%% Line
# Create fake data and boring plot
x = np.linspace(0, 20, 100)
y = np.sin(x)
plt.plot(x,y)
plt.xlabel('Explanatory')
plt.ylabel('Dependent')
# plt.title('Example Data')  # this code returns output with it ("Text(0.5 ....)") - to get rid of this, end the line with a semicolon
plt.title('Example Data');
# plt.show # this isn't important here because we are running this .py file like a .ipynb file


' Lesson 15. Scatterplot '
# %%
# Simple Scatter
xy = np.random.randn(100,2)
plt.scatter(xy[:,0], xy[:,1]);

# %%
X = np.random.randn(200,2)
X[:50] += 3
Y = np.zeros(200)
Y[:50] = 1

plt.scatter(X[:,0], X[:,1], c=Y)

' Lesson 16. Histogram'
# %%
X = np.random.randn(10000)
plt.hist(X, bins=50);
print("Mean of X={}".format(X.mean().round(2)))
print("SD of X={}".format(X.std().round(2)))

' Lesson 17. Plotting Images'
# %%
# you can view images using the PIL package
# then converting to a np.array
# plotting these images (either the original or the numpy array) is simple
    # plt.imshow(image)
    # image will be a 3 channel object with the third channel representing the color
    # if we smash the third channel down, it results in an image that is green...this is a mapping done by Python (how it interprets the lack of third channel)
    # plt.imshow(cmap='gray') will make the smashed image into black and white

' Lesson 18. Matplotlib Exercise '
x = np.random.random((10000,2)) * 2 - 1
y = np.zeros(10000)

y[np.sign(x[:,0]) == np.sign(x[:,1])] = 2
print(x)
print(y)

plt.scatter(x[:,0], x[:,1], c=y);

# %%
