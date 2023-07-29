import numpy as np
# We intend to create a dataframe which will be saved and used for regression.

def count_sixes(test_size):
    """Takes an integer and rolls a dice test_size times, 
    before returning the number of 6s rolled."""
    number_sixes=0
    for i in range(test_size):
        if np.random.randint(1,7)==6:
            number_sixes=number_sixes+1
    return number_sixes

def make_distribution(test_size,ntests,as_pct=False):
    """Returns a distribution (array) where each entry represents
     the outcome of a test (with ntests in all), and each test consists
     of rolling a die test_size times and counting the number of sixes obtained.
     If as_pct=True, then the entries represent the percent (as a decimal)
     of sixes obtained out of the total rolls in the corresponding test."""
    if as_pct==False:
        return np.array([count_sixes(test_size) for i in range(ntests)])
    elif as_pct==True:
        return np.array([count_sixes(test_size)/test_size for i in range(ntests)])
    else:
        print("There was an issue with the argument as_pct.")

def add_three(a,b,c=5):
    return a+b+c


print("This should print when the module is imported.")

# We'll see if this works, and save a sample DataFrame.  
# The idea is as follows.  Given a random collection of 1000 pairs of the form (test_size,ntests),
# We will input these pairs into make_distribution(as_pct=True) to obtain a collection of 1000 different distributions.
# Each row of the dataframe will contain information about one of the distributions (there will be 1000 rows).
# The dataframe will consist of four columns:
# 'Rolls per Test': how many times the dice are rolled per test.
# 'Number of Tests': how many tests are performed (how long the distribution is.)
# 'Mean': The mean of the values in the distribution.
# 'std': The standard deviation of the values in the distribution.
# 'Median': The median of the values in the distribution.
# 'IQR': The interquartile range (the size of the interval between the first and third quartile) of the values in the distribution.
if __name__ == "__main__":
    print("This should not print when the module is imported.")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import time
    np.random.seed(1)
    X = make_distribution(100,1000)
    print(X)
    sns.set_style("darkgrid")
    sns.kdeplot(X)
    plt.xlabel("Number of sixes rolled in test")
    plt.title("X")
    plt.show()
    print(len(X))
    # This works for creating a single distribution.  Let's make a dataframe.

    df = pd.DataFrame({'Rolls per Test': [np.random.randint(10,500) for i in range(1000)]})
    df['Number of Tests']=[np.random.randint(50,1000) for i in range(1000)]
    df.info()
    # For various reasons we would prefer not to have any duplicate pairs (fully duplicate rows).
    # We will fix this.
    duplicated = df[df.duplicated()]
    print("Before adjusting cells the set of duplicate rows is:")
    print(duplicated)
    while duplicated.shape[0]>0:
        for i in duplicated.index:
            df.at[i, 'Number of Tests']=np.random.randint(50,1000)
        duplicated=df[df.duplicated()]
    print("After adjusting the set of duplicate rows is:")
    print(df[df.duplicated()])
    print("Our dataframe now looks like:")
    df.info()
    print("Generating the distributions will take a long time.")
    start= time.time()
    df['X']=df.apply(lambda row : make_distribution(row['Rolls per Test'],
                                                    row["Number of Tests"],
                                                    as_pct=True), axis=1)
    stop=time.time()
    print("Generating the distributions took about {} minutes.".format(round((stop-start)/60,2)))
    print(df.head())
    df['Mean'] = [np.mean(X) for X in df['X']]
    df['std'] = [np.std(X) for X in df['X']]
    df['Median'] = [np.median(X) for X in df['X']]
    df['IQR'] = [np.quantile(X,.75)-np.quantile(X,.25) for X in df['X']]
    df.info()
    print(df)
    df=df[['Rolls per Test','Number of Tests', 'Mean', 'std', 'Median', 'IQR']]
    df.info()
    df.to_csv('regression-data.csv',index=False)
