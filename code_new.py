import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools as iter
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from numpy import arange
from scipy import stats
import math

df0 =pd.read_csv("/desktop/data.csv")

df0.head()
df = df0.T
df = df.drop('Country Code')
df = df.drop('Indicator Name')
df = df.drop('Indicator Code')
df = df.drop('1960')
new_header = df.iloc[0] #grab the first row for the header
df = df[1:] #take the data less the header row
df.columns = new_header #set the header row as the df header


def popplot (indata):

  """
    This function is used to plot the population growth in %. Pre processed data
    from the previous function is passed here and the graphs are plot
    """

  plt.figure(figsize=(10,6))
  plt.scatter(indata['United Arab Emirates'],indata['South Africa'])
  plt.xlabel('Population growth (%) of United Arab Emirates from 1961 to 2021')
  plt.ylabel('Population growth (%) of South Africa from 1961 to 2021')
  plt.title('Population growth (%) - UAE vs SA')

def clusterscore (Xin):

  """
    This function is used to plot Elbow graph to identify the optimum
    number of clusters. Pre selected value from the raw data is passed to 
    this function and the plot is returned.
    """

  clustering_score = []
  for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'random', random_state = 42)
    kmeans.fit(Xin)
    clustering_score.append(kmeans.inertia_) # inertia_ = Sum of squared distances of samples to their closest cluster center.
    
  plt.figure(figsize=(10,6))
  plt.plot(range(1, 11), clustering_score)
  plt.scatter(5,clustering_score[4], s = 200, c = 'red', marker='*')
  plt.title('The Elbow Method')
  plt.xlabel('No. of Clusters')
  plt.ylabel('Clustering Score')
  plt.show()

def kclustering(n_clusters):

  """
    This function is used to plot the predicted clusters. No of 
    clusters is passed to this function based on the elbow graph
    genderated in previous function. The graph with clusters is 
    returned from this function.
    """

  kmeans= KMeans(n_clusters, random_state = 42)
  # Compute k-means clustering
  kmeans.fit(X)
  # Compute cluster centers and predict cluster index for each sample.
  pred = kmeans.predict(X)
  plt.figure(figsize=(10,6))
  plt.scatter(X[pred == 0, 0], X[pred == 0, 1], c = 'brown', label = 'Cluster 0')
  plt.scatter(X[pred == 1, 0], X[pred == 1, 1], c = 'green', label = 'Cluster 1')
  plt.scatter(X[pred == 2, 0], X[pred == 2, 1], c = 'blue', label = 'Cluster 2')
  plt.scatter(X[pred == 3, 0], X[pred == 3, 1], c = 'purple', label = 'Cluster 3')
  plt.scatter(X[pred == 4, 0], X[pred == 4, 1], c = 'orange', label = 'Cluster 4')

  plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1],s = 300, c = 'red', label = 'Centroid', marker='*')

  plt.xlabel('Population growth (%) of United Arab Emirates from 1961 to 2021')
  plt.ylabel('Population growth (%) of South Africa from 1961 to 2021')

  plt.legend()
  plt.title('Population growth Clusters')


popplot(df) # to plot the population growth % - UAE vs SA
X = df.iloc[:, [8,263]].values # used to select a particular set of data
clusterscore(X)# to plot the elbow plot
no_of_clusters = 5
kclustering(no_of_clusters) # to generate the graph with clusters displayed

#-----End of clustering segment----
#----Starting of Curve fitting segment----

x = list(range(1961,2022))
y = df['South Africa']

# define the true objective function
def objective(x, a, b, c, d, e, f):
  return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + (f)

plt.figure(figsize=(10,6))
# curve fit
popt, covar = curve_fit(objective, x, y)

# summarize the parameter values
a, b, c, d, e, f = popt

# define a sequence of inputs between the smallest and largest known inputs
x_line = arange(min(x), max(x), 1)

# calculate the output for the range
y_line = objective(x_line, a, b, c, d, e, f)

def err_ranges(x, func, param, sigma):

    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper 


def bestfitcurve (x,y,x_line,y_line):
  """
    This function plots the best fit curve, lower and upper are given
    as function inputs and the curve is generated as the output
    """
    
  # plot input vs output
  plt.scatter(x, y)
  # create a line plot for the mapping function
  plt.plot(x_line, y_line, '-', color='red')
  ci = 0.1
  plt.fill_between(x_line, (y_line-ci), (y_line+ci), color='green', alpha=0.4)
  plt.xlabel('Years')
  plt.ylabel('Population growth (%) in South Africa ')
  plt.title('Population growth (%) in South Africa from 1960 to 2021 ')
  plt.show()

bestfitcurve (list(range(1961,2022)), df['South Africa'],x_line,y_line)
