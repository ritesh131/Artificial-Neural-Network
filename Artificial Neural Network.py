#!/usr/bin/env python
# coding: utf-8

# ### Artificial Neural Network using Keras

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv(r'DataSet/data.csv')
data.head()


# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


data.hist(bins=50,figsize=(20,15))
plt.show()


# In[6]:


ax = sns.countplot(data['diagnosis'], label='Count')
B, M = data['diagnosis'].value_counts()
print('Benign', B)
print('Malignanat', M)


# In[7]:


del data['Unnamed: 32']
data.head()


# ### Data PreProsesing

# In[8]:


X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ### Import all require library to build ANN Model

# In[9]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


# In[10]:


# Initialising the ANN
classifier = Sequential()


# In[11]:


# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=16, init='uniform', activation='relu', input_dim=30))
# Adding dropout to prevent overfitting
classifier.add(Dropout(p=0.1))


# In[12]:


# Adding the second hidden layer
classifier.add(Dense(output_dim=16, init='uniform', activation='relu'))
# Adding dropout to prevent overfitting
classifier.add(Dropout(p=0.1))


# In[13]:


# Adding the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))


# In[14]:


# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[15]:


# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=100, nb_epoch=150)
# Long scroll ahead but worth
# The batch size and number of epochs have been set using trial and error. Still looking for more efficient ways. Open to suggestions.


# In[16]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# In[17]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[18]:


print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/57)*100))


# In[19]:


sns.heatmap(cm,annot=True)
plt.savefig('h.png')

