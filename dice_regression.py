import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

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
plt.savefig('images/rolls-per-test.png')
plt.show()

# We now repeat with the Number of Tests as the x axis.
fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
for ax,col in zip(axes.flatten(),df.columns):
    sns.scatterplot(data=df, x='Number of Tests', y=col, ax=ax)
plt.tight_layout()
plt.savefig('images/number-of-tests.png')
plt.show()

# It is clear that the size of each test is more closely correlated with results closer to ~16%, and generally precise results.
  
print(df.corr())
# Let us start performing regression.  Note that the correlation between the rolls per test and the IQR is fairly high 
# (in the negative direction) and from the graph appears to demonstrate something like exponential decay.  
# Let us try a few different regression models for this data.
# First, let us split the data into a training and a test portion using scikitlearn.

train_df, test_df = train_test_split(df,train_size=.8, random_state=2)
print("We have split into a training and testing dataframe. Let's examine the training dataframe.")
train_df.info()
print(train_df)
print("Now let us examine the test dataframe.")
test_df.info()
print(test_df)

# We will check that the training data seems relatively similar to the overall set.
plt.figure(figsize=(6,4))
sns.scatterplot(data=train_df, x='Rolls per Test', y='IQR', label='Training')
sns.scatterplot(data=test_df, x='Rolls per Test', y='IQR',label='Testing')
plt.title('Comparing testing/training')
plt.legend()
plt.show()
# The training data looks good.  Let us start regression analysis.
########################################################################
# First let us try polynomial regression.  It looks like a linear or quadratic function would be unlikely to fit
# the curve, so we will try a third degree polynomial: y=ax^3+bx^2+cx+d with a,b,c,d real numbers.

cubic_fit=np.polynomial.Polynomial.fit(train_df['Rolls per Test'],train_df['IQR'],3).convert()
print(cubic_fit)

plt.figure(figsize=(8,6))
sns.scatterplot(data=train_df, x='Rolls per Test', y='IQR', label='Actual (Training)')
sns.lineplot(x=train_df['Rolls per Test'],y=cubic_fit(train_df['Rolls per Test']), color='C1', label='Predicted')
plt.title('Cubic model')
plt.legend()
plt.show()

print('For the cubic model:')
print('The Coefficient of Determination on the training data is', r2_score(train_df['IQR'],cubic_fit(train_df['Rolls per Test']))) 

# Let us see how the model works for the testing data.
plt.figure(figsize=(8,6))
sns.scatterplot(data=test_df, x='Rolls per Test', y='IQR',label='Actual (Testing)')
sns.lineplot(x=test_df['Rolls per Test'],y=cubic_fit(test_df['Rolls per Test']), color='C1', label='Predicted')
plt.title('Cubic model')
plt.legend()
plt.show()
print('The Coefficient of Determination on the testing data is', r2_score(test_df['IQR'],cubic_fit(test_df['Rolls per Test'])))
# Note that the R^2 value is fairly good, but it looks like there might be overfitting based on the graph. 
# Can we do better?  Let us consider a different kind of model.
########################################################################
# Next let us try an exponential model.
# If x is the number of rolls per test, and y is the corresponding IQR, we should try to fit the model y=a(e^(kx)) where
# a and k are constants.
# taking a logarithm on each side we obtain log(y)=log(a)+kx.
# If we perform linear regression on the pair (x,log(y)) we should obtain a slope (to approximate k) 
# and an intercept (to approximate log(a)).
exp_fit = np.polynomial.Polynomial.fit(train_df['Rolls per Test'],np.log(train_df['IQR']),1).convert()
print(exp_fit)

def exp_model(x):
    """Returns the y value for a given x input in our exponential model"""
    # y=a(e^(kx))
    # Here k=fit[1] and a=exp(fit[0]) (since log(a)=fit[0])  
    return np.exp(exp_fit.coef[0])*np.exp(x*exp_fit.coef[1])

plt.figure(figsize=(8,6))
sns.scatterplot(data=train_df, x='Rolls per Test', y='IQR', label='Actual (Training)')
sns.lineplot(x=train_df['Rolls per Test'],y=exp_model(train_df['Rolls per Test']), color='C1', label='Predicted')
plt.title('Exponential model')
plt.legend()
plt.show()

print('For the exponential model:')
print('The Coefficient of Determination on the training data is', r2_score(train_df['IQR'],exp_model(train_df['Rolls per Test']))) 

# Let us see how the model works for the testing data.
plt.figure(figsize=(8,6))
sns.scatterplot(data=test_df, x='Rolls per Test', y='IQR',label='Actual (Testing)')
sns.lineplot(x=test_df['Rolls per Test'],y=exp_model(test_df['Rolls per Test']), color='C1', label='Predicted')
plt.title('Exponential model')
plt.legend()
plt.show()

print('The Coefficient of Determination on the testing data is', r2_score(test_df['IQR'],exp_model(test_df['Rolls per Test']))) 

##########################################################
# Let us try a power model of the form y=k(x^a) where k and a are real constants.
# Applying log to each side, we have log(y)=log(k)+alog(x).
# We can perform regression on the (input,output) pair (log(x),log(y)) to obtain a slope a and intercept log(k).
power_fit = np.polynomial.Polynomial.fit(np.log(train_df['Rolls per Test']),np.log(train_df['IQR']),1).convert()
print(power_fit)

def power_model(x):
    """Returns the y value for a given x input in our power model"""
    # y=k(x^a)
    # Here a=fit[1] and k=exp(fit[0]) (since log(k)=fit[0])  
    return np.exp(power_fit.coef[0])*(x**power_fit.coef[1])

plt.figure(figsize=(8,6))
sns.scatterplot(data=train_df, x='Rolls per Test', y='IQR', label='Actual (Training)')
sns.lineplot(x=train_df['Rolls per Test'],y=power_model(train_df['Rolls per Test']), color='C1', label='Predicted')
plt.title('Power model')
plt.legend()
plt.show()
print('For the power model:')
print('The Coefficient of Determination on the training data is', r2_score(train_df['IQR'],power_model(train_df['Rolls per Test']))) 

# Let us see how the model works for the testing data.
plt.figure(figsize=(8,6))
sns.scatterplot(data=test_df, x='Rolls per Test', y='IQR',label='Actual (Testing)')
sns.lineplot(x=test_df['Rolls per Test'],y=power_model(test_df['Rolls per Test']), color='C1', label='Predicted')
plt.title('Power model')
plt.legend()
plt.show()

print('The Coefficient of Determination on the testing data is', r2_score(test_df['IQR'],power_model(test_df['Rolls per Test']))) 
# This looks, both from the R^2 score and the graph, like the most accurate choice of regression for the dataset.
# Let us save a comparison of the different regression curves.

plt.figure(figsize=(8,6))
sns.set_style('darkgrid')
sns.scatterplot(data=test_df, x='Rolls per Test', y='IQR',label='Actual', alpha=.75)
sns.lineplot(x=test_df['Rolls per Test'],y=cubic_fit(test_df['Rolls per Test']), color='C1', label='Cubic', linewidth=2)
sns.lineplot(x=test_df['Rolls per Test'],y=exp_model(test_df['Rolls per Test']), color='C2', label='Exponential', linewidth=2)
sns.lineplot(x=test_df['Rolls per Test'],y=power_model(test_df['Rolls per Test']), color='C3', label='Power', linewidth=2)
plt.title('Comparing models')
plt.legend()
plt.savefig('images/comparing-models.png')
plt.show()