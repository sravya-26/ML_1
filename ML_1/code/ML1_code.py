#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sklearn.datasets import load_svmlight_files

X_train, y_train, X_test, y_test = load_svmlight_files(("C:\\Users\\sravy\\Downloads\\Applied ML\\usps\\usps", "C:\\Users\\sravy\\Downloads\\Applied ML\\usps.t\\usps.t"))
X_train, X_test = X_train.toarray(), X_test.toarray()

train_data, train_labels = load_svmlight_file("C:\\Users\\sravy\\Downloads\\Applied ML\\usps\\usps")
test_data, test_labels = load_svmlight_file("C:\\Users\\sravy\\Downloads\\Applied ML\\usps.t\\usps.t")

train_data = train_data.toarray()
test_data = test_data.toarray()


# In[3]:


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# In[4]:


def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))


# In[5]:


def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# In[6]:


def find_nearest_neighbor(test_vector, X_train, distance_metric):
    distances = [distance_metric(test_vector, train_point) for train_point in X_train]
    return np.argmin(distances)


# In[7]:


def evaluate(X_test, y_test, X_train, y_train, distance_metric):
    correct = 0
    for i, test_vector in enumerate(X_test[:10]):  
        nearest_idx = find_nearest_neighbor(test_vector, X_train, distance_metric)
        if y_test[i] == train_labels[nearest_idx]:
            correct += 1
    return correct / 10 * 100


# In[8]:


euclidean_acc = evaluate(X_test, y_test, X_train, y_train, euclidean_distance)
manhattan_acc = evaluate(X_test, y_test, X_train, y_train, manhattan_distance)
cosine_acc = evaluate(X_test, y_test, X_train, y_train, cosine_distance)

print(f"Euclidean Accuracy: {euclidean_acc}%")
print(f"Manhattan Accuracy: {manhattan_acc}%")
print(f"Cosine Accuracy: {cosine_acc}%")


# In[ ]:





# In[ ]:




