# Phase 4 Code Challenge Review

TOC:

  - [PCA](#pca)
  - [NLP](#nlp)
  - [Time Series](#ts)  
  - [Clustering](#clust)


<a id='pca'></a>

# PCA

When creating principle components, PCA aims to find a vectors in the direction of our feature space that is fit to what?

Principal Component Analysis creates a set of features called principal compenents. This reduces the dimensions of our data set from the original `n` components.  The components are built successively.  Describe what the first principal component represents in relation to the original feature set.

> Your answer here

==SOLUTION==

The first principal component points in the direction that explains the most variance of the original feature set.  Each principal component is composed of a combination of the original components. A larger weight in first principal component indicates a larger variance in the original feature set.

Why is scaling important for PCA?

> Your answer here

==SOLUTION==

Scaling is important because variance in a feature measured in a relatively small unit can be just as or more important than a feature measured in a large unit.  In other words, the dependent variable may depend more on the feature with the smaller unit than the large.  When transforming a dataset with PCA, the PCA object finds the direction that explains the most total variance in the feature set.  It will then tend to identify features with larger units as the most important.  By scaling, the unit is taken out of the picture.  PCA will be able to identify features whose original scale is smaller, but whose variation correlates more closely with the dependent feature.  

Take for example a model that attempts to predict weight of a new born with age of the mother in years and height of the mother in meters.  Without scaling, the total variance of age across all subjects will be much greater than the total variance of height, simply because of the unit.  If one fits a PCA object to height and age, the first principal component will be more heavily influenced by age.  However, the height of the mother likely is a better predictor than age, but because of its relatively small variance in the original unit (meters), its influence is obscured. By scaling, PCA will consider the relative variance of height and age without regard for the unit. 


What are some reasons for using PCA?


> Your answer here

==SOLUTION==

1. PCA can speed up computation time. 
2. PCA can help with overfitting and decrease the overall prediction error of the model.
3. A similar advantage to #2 is that PCA eliminates multicollinearity.  Because each component is built orthogonally to the last, all multicollinearity is eliminated.  
4. PCA can be used for visualization.  Reducing the original data set to two dimensions allows a representation of the data to be plotted on an x-y plane.


How can one determine how many principle components to use in a model?

> Your answer here

# Now let's implement PCA in code.


```python
import pandas as pd
from sklearn.datasets import  load_breast_cancer
data = load_breast_cancer()
X = pd.DataFrame(data['data'], columns = data['feature_names'])
X.head()
```


```python
# appropriately preprocess X
```


```python
# instantiate a pca object with 2 components
```


```python
# determine how much of the total variance is explained by the first two components
```

<a id='nlp'></a>


# NLP

For NLP data, what is the entire data of records called?

> your answer here

What is an individual record called?

> your answer here

What is a group of two words that appear next to one-another in a document?

> Your answer here

What is a high frequency, semantically low value word called? 

> Your answer here

List the preprocessing steps we can employ to create a cleaner feature set to our models.

> Your answer here

Explain the difference between the two main vectorizors we employ to transform the data into the document-term matrix.

> Your answer here

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

policies.head()
```


```python
# split into train and test set 
# note: for demonstration purposes, we will not use cross-validation here nor a holdout set.
```


```python
# Import and instantiate a Count Vectorizer with defaults
```


```python
# Transform train and test sets with the Count Vectorizer
# then fit a logistic regression model on it.
```


```python
# Score on both train and test sets.
```


```python
# Tune some hyperparameters of the vectorizer and assess the performance
```

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


```python
# Perform that step in code
```

What types of patterns might we expect to find in our time series datasets?


```python
# plot the time series
```

What type of patterns do you see in the above plot?

> Your answer here


```python
# Add to the plot to visualize patterns by looking at summary statistics across a window of time.
```

What are some ways to remove those trends? 

What is the goal of removing those trends?

> Your answer here


```python
# Attempt to make the series stationary using differencing
```

How can we diagnose whether we have successfully removed the trends?

> Your answer here

Use the Augmented Dickey Fuller test to see if the detrended data is ready for modeling


```python
# your code here
```

<a id='clust'></a>

# Clustering

Question: What is the difference between supervised and unsupervised learning?

> Your answer here

Describe how the kmeans algorithm updates its cluster centers after initialization.

> Your answer here

What is inertia, and how does kmeans use inertia to determine the best estimator?

> Your answer here


```python
from sklearn.cluster import KMeans

KMeans()
```

What other metric do we have to score the clusters which are formed?

Describe the difference between it and inertia.

> Your answer here

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

# Bonus: Use PCA to visualize in two dimensions the cluster groups of the best metric.


```python
# your code here
```
