# Phase 4 Code Challenge Review

TOC:

  - [PCA](#pca)
  - [NLP](#nlp)
  - [Time Series](#ts)  
  - [Clustering](#clust)



```python
# Students, ignore this cell
%load_ext autoreload
%autoreload 2

import sys
sys.path.append('../../..')

from new_caller.random_student_engager.student_caller import CohortCaller
from new_caller.random_student_engager.student_list import avocoder_toasters

caller = CohortCaller(avocoder_toasters)
```

    hello


<a id='pca'></a>

# PCA

When creating principle components, PCA aims to find a vector in the direction of our feature space that is fit to what?

> Your answer here


```python
caller.call_n_students(1)
```

Why is scaling important for PCA?

> Your answer here


```python
caller.call_n_students(1)
```

What are some reasons for using PCA?


> Your answer here


```python
caller.call_n_students(1)
```

How can one determine how many principle components to use in a model?

> Your answer here


```python
caller.call_n_students(1)
```

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


```python
caller.call_n_students(1)
```

What is an individual record called?

> your answer here


```python
caller.call_n_students(1)
```

What is a group of two words that appear next to one-another in a document?

> Your answer here

What is a high frequency, semantically low value word called? 

> Your answer here


```python
caller.call_n_students(1)
```

List the preprocessing steps we can employ to create a cleaner feature set to our models.

> Your answer here


```python
caller.call_n_students(1)
```

Explain the difference between the two main vectorizors we employ to transform the data into the document-term matrix.

> Your answer here


```python
caller.call_n_students(1)
```

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


```python
caller.call_n_students(1)
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


```python
caller.call_n_students(1)
```

What types of patterns might we expect to find in our time series datasets?


```python
# plot the time series
```


```python
caller.call_n_students(1)
```

What type of patterns do you see in the above plot?

> Your answer here


```python
caller.call_n_students(1)
```


```python
# Add to the plot to visualize patterns by looking at summary statistics across a window of time.
```


```python
caller.call_n_students(1)
```

What are some ways to remove those trends? 


```python
caller.call_n_students(1)
```

What is the goal of removing those trends?

> Your answer here


```python
# Attempt to make the series stationary using differencing
```


```python
caller.call_n_students(1)
```

How can we diagnose whether we have successfully removed the trends?

> Your answer here


```python
caller.call_n_students(1)
```

Use the Augmented Dickey Fuller test to see if the detrended data is ready for modeling


```python
# your code here
```


```python
caller.call_n_students(1)
```

<a id='pca'></a>

<a id='clust'></a>

# Clustering

Question: What is the difference between supervised and unsupervised learning?

> Your answer here


```python
caller.call_n_students(1)
```

Describe how the kmeans algorithm updates its cluster centers after initialization.

> Your answer here


```python
caller.call_n_students(1)
```

What is inertia, and how does kmeans use inertia to determine the best estimator?

> Your answer here


```python
from sklearn.cluster import KMeans

KMeans()
```


```python
caller.call_n_students(1)
```

What other metric do we have to score the clusters which are formed?

Describe the difference between it and inertia.

> Your answer here


```python
caller.call_n_students(1)
```

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


```python

```
