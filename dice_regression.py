import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df=pd.read_csv('regression-data.csv')
df.info()
print(df.columns)
# We create a figure and six axes in a 2x3 grid.  We will have them share the same x values.
fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)

# In the following line we iterate through pairs of axes and column names of the dataframe.  We use flatten() to turn into a 
# one dimensional array to fit our needs.  
for ax,col in zip(axes.flatten(),df.columns):
    sns.scatterplot(data=df, x='Rolls per Test', y=col, ax=ax)
plt.tight_layout()
plt.show()

# We now repeat with the Number of Tests as the x axis.
fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
for ax,col in zip(axes.flatten(),df.columns):
    sns.scatterplot(data=df, x='Number of Tests', y=col, ax=ax)
plt.tight_layout()
plt.show()

# It is clear that the size of each test is more closely correlated with results closer to ~16%, and generally precise results.
  
print(df.corr())
# Let us start performing regression.  Note that the correlation between the rolls per test and the IQR is fairly high 
# (in the negative direction) and from the graph appears to demonstrate exponential decay.  Let us try to create an exponential
# regression model for this data.
# If x is the number of rolls per test, and y is the corresponding IQR, we should try to fit the model y=a(e^(kx)) where
# a and k are constants.
# taking a logarithm on each side we obtain log(y)=log(a)+kx.
# If we perform linear regression on the pair (x,log(y)) we should obtain an intercept (log(a)) and a slope (k).
fit = np.polyfit(df['Rolls per Test'],np.log(df['IQR']),1)
print(fit)

def expmodel(x):
    return np.exp(fit[1])*np.exp(x*fit[0])

plt.figure(figsize=(8,6))
sns.lineplot(x=df['Rolls per Test'],y=expmodel(df['Rolls per Test']))
sns.scatterplot(data=df, x='Rolls per Test', y='IQR')
plt.show()