#### Code for Section 5 - Scipy of the Deep Learning Prerequisities course

# Scipy is a package to get access to the normal statistical distributions
    # along with corresponding values such as PDF and CDF
#%%
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

#%%

' Lesson 28. PDF and CDF'
x = np.linspace(-5,5,1000)

# standard normal 
fx = norm.pdf(x, loc = 0, scale = 1)

plt.plot(x, fx)
plt.xlabel("x")
plt.ylabel('Density')
plt.title("PDF of Standard Normal");
# %%
# CDF of Standard Normal
Fx = norm.cdf(x)
plt.plot(x, Fx)
plt.xlabel("x")
plt.ylabel('Density')
plt.title("CDF of Standard Normal");

# %%
# Log PDF of Standard Normal
Fx = norm.logpdf(x)
plt.plot(x, Fx)
plt.xlabel("x")
plt.ylabel('Density')
plt.title("Log PDF of Standard Normal");


