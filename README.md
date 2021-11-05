# Phase 4 Code Challenge Review

TOC:

  - [PCA](#pca)
  - [NLP](#nlp)
  - [Time Series](#ts)  
  - [Clustering](#clust)


<a id='pca'></a>

# PCA

Principal Component Analysis creates a set of features called principal compenents. PCA reduces the dimensions of our data set from the original n number of features to a specified number of components.  

Describe what the first principal component represents in relation to the original feature set.

> Your answer here


The first principal component points in the direction that explains the most variance of the original feature set.  Each principal component is composed of a combination of the original components. A larger weight in the first principal component indicates a feature with a larger variance in the original feature set.

Why is scaling important for PCA?

> Your answer here


Scaling is important because variance in a feature measured in a relatively small unit can be just as or more important than a feature measured in a large unit.  In other words, the dependent variable may depend more on the feature with the smaller unit than the large.  When transforming a dataset with PCA, the PCA object finds the direction that explains the most total variance in the feature set.  It will then tend to identify features with larger units as the most important.  By scaling, the unit is taken out of the picture.  PCA will be able to identify features whose original scale is smaller, but whose variation correlates more closely with the dependent feature.  

Take for example a model that attempts to predict weight of a new born with age of the mother in years and height of the mother in meters.  Without scaling, the total variance of age across all subjects will be much greater than the total variance of height, simply because of the unit.  If one fits a PCA object to height and age, the first principal component will be more heavily influenced by age.  However, the height of the mother likely is a better predictor than age, but because of its relatively small variance in the original unit (meters), its influence is obscured. By scaling, PCA will consider the relative variance of height and age without regard for the unit. 


What are some reasons for using PCA?


> Your answer here


1. PCA can speed up computation time. 
2. PCA can help with overfitting and decrease the overall prediction error of the model.
3. A similar advantage to #2 is that PCA eliminates multicollinearity.  Because each component is built orthogonally to the last, all multicollinearity is eliminated.  
4. PCA can be used for visualization.  Reducing the original data set to two dimensions allows a representation of the data to be plotted on an x-y plane.
5. PCA can make the data more interpretable. With high dimensional data, by looking at the weights of the different components, one can see which features have siginificant variance, and which don't. 

How can one determine how many principle components to use in a model?

> Your answer here

Each successive principal component explains a different aspect of the variance of the feature set.  After the PCA object has been fit, it has an attribute named explained_variance_ratio, which describes what percentage of the variance is explained by each component.  The first component will have the largest value for explained variance ratio.  And the variance explained will decrease with each successive component. At some point, adding another principal component will result in only a small additional percent of variance explained.  Looking at a graph of the number of components vs. cumulative explained variance ratio, the curve will stop increasing significantly and start to level off.  A good choice for the number of principal components would be the component after which a significant increase of explained variance stops.

The PCA object can also be given a proportion as the argument p_components.  PCA will stop after that decimal.  If .8 is given, PCA will stop adding components after the previous components have explained 80% of the variance.

# Now let's implement PCA in code.


```python
import pandas as pd
from sklearn.datasets import  load_breast_cancer
import matplotlib.pyplot as plt

data = load_breast_cancer()
X = pd.DataFrame(data['data'], columns = data['feature_names'])
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
# appropriately preprocess X
```


```python
# instantiate a pca object with 2 components and fit to preprocessed X
```


```python
# determine how much of the total variance is explained by the first two components
```


```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Scale the independent features
ss = StandardScaler()
ss.fit(X)

X_sc = ss.transform(X)

# instanstiate a pca object with 2 components
pca = PCA(n_components=2)

#fit to preprocessed X
pca.fit(X_sc)

print(sum(pca.explained_variance_ratio_))
print('The first two principal components explain about 63% of the variance')
```

    0.6324320765155944
    The first two principal components explain about 63% of the variance


<a id='nlp'></a>


# NLP


```python
# Each sentence is a document
sentence_one = "Harry Potter is the best young adult book about wizards"
sentence_two = "Um, EXCUSE ME! Ever heard of Earth Sea?"
sentence_three = "I only like to read non-fiction.  It makes me a better person."

# The corpus is composed of all of the documents
corpus = [sentence_one, sentence_two, sentence_three]
```

Describe what preproccessing steps to take in order to isolate the semantically valuable components of the corpus.

1. Remove punctuation.
2. Make all tokens lowercase.
3. Remove stop words. 
4. Stem and lemmatize. 

Describe the rows and columns of the document term matrix that would result from vectorizing the above corpus. 


In the document term matrix, each row would represent a sentence.  There would be as many columns as there are tokens, after preprocessing, accross the entire corpus.  


Describe what the values within the document-term matrix could be.


The values are some representation of frequency.  If a count vectorizor is used, the values are counts of how many times a given word occurs in a sentence. 

Describe how to calculate a TF-IDF score.


The term frequency component of TFIDF counts the number of times a term shows up in a given document. This number is multiplied by the inverse document frequency. IDF is calculated as the log of the total number of documents divided by the number of documents the term appears in. If the term appears in less documents, the IDF score will be larger. 

# NLP Code


```python
# data import
policies = pd.read_csv('data/2020_policies_feb_24.csv')

def warren_not_warren(label):
    
    '''Make label a binary between Elizabeth Warren
    speeches and speeches from all other candidates'''
    
    if label =='warren':
        return 1
    else:
        return 0
    
policies['candidate'] = policies['candidate'].apply(warren_not_warren)
```

The dataframe loaded above consists of policies of 2020 Democratic presidential hopefuls. The `policy` column holds text describing the policies themselves.  The `candidate` column indicates whether it was or was not an Elizabeth Warren policy.


```python
policies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>name</th>
      <th>policy</th>
      <th>candidate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>100% Clean Energy for America</td>
      <td>As published on Medium on September 3rd, 2019:...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>A Comprehensive Agenda to Boost America’s Smal...</td>
      <td>Small businesses are the heart of our economy....</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>A Fair and Welcoming Immigration System</td>
      <td>As published on Medium on July 11th, 2019:\nIm...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>A Fair Workweek for America’s Part-Time Workers</td>
      <td>Working families all across the country are ge...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>A Great Public School Education for Every Student</td>
      <td>I attended public school growing up in Oklahom...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



The documents for activity are in the `policy` column, and the target is candidate. 


```python
# split into train and test set with default arguments and random_state=42
# note: for demonstration purposes, we will not use cross-validation here nor a holdout set.
# note: whether you pass an array or a dataframe to as the feature set will change how it is
# passed to the vectorizer
```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(policies['policy'], policies['candidate'], 
                                                   random_state=42)
```


```python
# Import and instantiate a Count Vectorizer with defaults
```


```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

```


```python
# Transform train and test sets with the Count Vectorizer
# then fit a logistic regression model on it.
# if you get the AttributeError: 'str' object has no attribute 'decode', pass solver='liblinear'
```


```python
from sklearn.linear_model import LogisticRegression

cv.fit(X_train)
X_train_cv = cv.transform(X_train)
X_test_cv = cv.transform(X_test)

lr = LogisticRegression(solver='liblinear')
lr.fit(X_train_cv, y_train)
```




    LogisticRegression(solver='liblinear')




```python
# Score on both train and test sets.
```


```python
print(lr.score(X_train_cv, y_train))
print(lr.score(X_test_cv, y_test))
```

    1.0
    0.8958333333333334



```python
# Tune some hyperparameters of the vectorizer and assess the performance
```


```python
def htune_cv(cv, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
    
    '''Pass in a vectorizer with a set of hyperameters
    and print the train and test scores of a logistic
    model fit on a vectorized version of X.
    '''
    
    cv.fit(X_train)
    X_train_cv = cv.transform(X_train)
    X_test_cv = cv.transform(X_test)

    lr = LogisticRegression(solver='liblinear')

    lr.fit(X_train_cv, y_train)
    print(lr.score(X_train_cv, y_train))
    print(lr.score(X_test_cv, y_test))
```


```python
cv = CountVectorizer(stop_words='english')
htune_cv(cv)
```

    1.0
    0.8958333333333334



```python
cv = CountVectorizer(stop_words='english', max_features=100)
htune_cv(cv)
```

    1.0
    0.9166666666666666


<a id='ts'></a>

# Time Series


```python
import pandas as pd
import numpy as np
```


```python
ap = pd.read_csv('data/AirPassengers.csv')
```

With the data above, what is the first step in transforming it into data suitable for our time series models?

> Your answer here

Convert the index to a datetime index.


```python
# Perform that step in code
```


```python
ap.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Month</th>
      <th>#Passengers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1949-01</td>
      <td>112</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1949-02</td>
      <td>118</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1949-03</td>
      <td>132</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1949-04</td>
      <td>129</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1949-05</td>
      <td>121</td>
    </tr>
  </tbody>
</table>
</div>




```python
ap['Month'] = pd.to_datetime(ap['Month'])
ap.set_index('Month', inplace=True)
ap.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#Passengers</th>
    </tr>
    <tr>
      <th>Month</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1949-01-01</th>
      <td>112</td>
    </tr>
    <tr>
      <th>1949-02-01</th>
      <td>118</td>
    </tr>
    <tr>
      <th>1949-03-01</th>
      <td>132</td>
    </tr>
    <tr>
      <th>1949-04-01</th>
      <td>129</td>
    </tr>
    <tr>
      <th>1949-05-01</th>
      <td>121</td>
    </tr>
  </tbody>
</table>
</div>



What types of patterns might we expect to find in our time series datasets?

1. Increasing or descreasing mean over time. 
2. Variance is different across different time periods (homo/heteroskedacicity).
3. Covariance should not be a function of time.
4. Seasonality: Cycle of repeated pattern over a period: Week, month, year. 


```python
# plot the time series
```


```python
ap.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12464bf28>




    
![png](index_files/index_63_1.png)
    


What type of patterns do you see in the above plot?

> Your answer here

Increasing mean. 
Increasing variance. 
Seasonality with peak in the summer. 


```python
# Add to the plot to visualize patterns by looking at summary statistics across a window of time.
```


```python
fig, ax = plt.subplots()
ap.rolling(12).mean().plot(ax=ax)
ap.rolling(12).std().plot(ax=ax)
ap.plot(ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1249b22b0>




    
![png](index_files/index_68_1.png)
    


What are some ways to remove those trends? 

> Your answer here

1. Differencing: Subtracting the previous value from the current value.
2. Use statsmodels seasonal decompose. 
3. Subtracting the rolling mean. Log transform. Take the square root.  

Attempt to make the series stationary using differencing


```python
# your code here
```


```python
fig, ax = plt.subplots()
ap.diff().plot(ax=ax)
ap.diff().rolling(window=12).mean().plot(ax=ax)
ap.diff().rolling(window=12).std().plot(ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1255e4f98>




    
![png](index_files/index_74_1.png)
    



```python
fig, ax = plt.subplots()
ap.diff().diff().plot(ax=ax)
ap.diff().diff().rolling(12).mean().plot(ax=ax)
ap.diff().diff().rolling(12).std().plot(ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x125840978>




    
![png](index_files/index_75_1.png)
    


How can we diagnose whether we have successfully removed the trends?

> Your answer here


Visually: we can plot the rolling mean and standard deviation along with the differenced time series.
Analytically: we can run an augmented Dickey Fuller test to check for trend over time.   If we reject the null hypothesis, we reject that the series is non-stationary.

Use the Augmented Dickey Fuller test to see if the detrended data is ready for modeling


```python
# your code here
```


```python
from statsmodels.tsa.stattools import adfuller

adfuller(ap.diff().diff().dropna())

# note: dickey fuller test does not check for variance or seasonality 
# Although we can reject the null hypothesis, we know there is still increasing variance 
# and seasonality present.
# [stack_overflow](https://stats.stackexchange.com/questions/131054/does-stationarity-under-adf-test-imply-mean-variance-and-covariance-stationary)
```




    (-16.384231542468488,
     2.7328918500143186e-29,
     11,
     130,
     {'1%': -3.4816817173418295,
      '5%': -2.8840418343195267,
      '10%': -2.578770059171598},
     988.6020417275604)



<a id='clust'></a>

# Clustering

Describe how the kmeans algorithm updates its cluster centers after initialization.

> Your answer here

For every point, KMeans measures the distance to every cluster center. It then assigns each point to the closest cluster center.  For each group of assigned points, Kmeans then computes the means of all points assigned to each cluster, and moves the cluster centers to those means.   

What is inertia, and how does kmeans use inertia to determine the best estimator?

> Your answer here


```python
# look at the docstring for help
from sklearn.cluster import KMeans

KMeans()
```




    KMeans()




Inertia is a statistic describing how tightly clustered a group of points are.  It is calculated by taking all points in a given cluster, finding the sum of squared distance from the centroid, and summing the results across all clusters.

KMeans uses inertia to find optimal cluster centers.  Since there is randomness built into the alogorithm's choice of initial cluster centers, the final centroids will be different across different initializations.  When sklearn's KMeans algo is fit, it runs it a set number of times (see n_init=10 in the docstring), and chooses the run with the lowest inertia as the final model.


What other metric do we have to score the clusters which are formed?

Describe the difference between it and inertia.

> Your answer here


Silhouette score is a measure that describes how similar points are in a cluster to the other points in its cluster, and how dissimilar the points are in a cluster from the next nearest cluster. Unlike inertia, a higher silhouette score is better, and the scores range from -1 to 1. 

# Code Cluster Practice with Heirarchical Agglomerative Clustering

After the above conceptual review of KMeans, let's practice coding with agglomerative clustering.


```python
from sklearn.datasets import load_iris

data = load_iris()
X = pd.DataFrame(data['data'])
y = data['target']
```


```python
# Import the relevent clusterer and instantiate an instance of it. 
# Indicate the number of clusters you want
```


```python
# Preprocess the data
```


```python
# Fit the object
```


```python
# Calculate a silhouette score
```


```python
# Repeat with another choice for number of clusters
```


```python
# Determine which is a better number
```


```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

agg = AgglomerativeClustering(n_clusters=3)

ss = StandardScaler()
X_sc = ss.fit_transform(X)

agg.fit(X_sc)

silhouette_score(X_sc, agg.labels_)
```




    0.4466890410285909




```python
agg = AgglomerativeClustering(n_clusters=4)

ss = StandardScaler()
X_sc = ss.fit_transform(X)

agg.fit(X_sc)

silhouette_score(X_sc, agg.labels_)

# The model with 3 clusters has a higher silhouette score, so is a better choice.
```




    0.4006363159855973


