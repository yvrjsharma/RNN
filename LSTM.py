#!/usr/bin/env python
# coding: utf-8

# # **IMDB Modelling Task**
# ## **RNN Variants** 

# Importing the required libraries

# In[ ]:


import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,LSTM, Embedding, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, SimpleRNN, Concatenate
from tensorflow.keras.datasets import imdb
import tensorflow as tf
#import kaggle
import pandas as pd
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.models import Model    
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model

#Employing Early stopping as a means of Validation
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy
import tensorflow_datasets as tfds
import nltk
nltk.download('stopwords')
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

#Employing Early stopping as a means of Validation
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

#Library for saving of a model
get_ipython().system(' pip install h5py')


# #### Getting data using Kaggle API
# #### This dataset is used for some building some models along with the data available via tensorflow_datasets. Following lines of code sets up Kaggle.json file. This file the examiner wil have to have on his system and it can be downloaded through Kaggle website. Details on how to download this file and how to set it up are available in the Appendix of the Report submitted.

# In[ ]:


get_ipython().system(' pip install -q kaggle')
from google.colab import files
files.upload()


# In[ ]:


# Choose the kaggle.json file that you downloaded
get_ipython().system(' mkdir ~/.kaggle')
get_ipython().system(' cp kaggle.json ~/.kaggle/')
#	Make directory named kaggle and copy kaggle.json file there.
get_ipython().system(' chmod 600 ~/.kaggle/kaggle.json')
#	Change the permissions of the file.
get_ipython().system(' kaggle datasets list')


# ### Importing IMDB data from Keras
# #### This dataset comes segmented as 25000 instances for training and 25000 testing. First merging the training and test segments and then resegmenting into Training-50%, Validation-30%, and Testing-20% as asked. 

# In[ ]:


from tensorflow.keras.datasets import imdb
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

data.shape, targets.shape
#data here represents the review variable, while targets refers to sentiment variable


# In[ ]:


#One loook at these two variables, data and targets
print(targets[0:10],data[0:10]) 


# ### Exploring the dataset

# In[ ]:


#Using the Keras IMDB dataset 
print("Categories:", np.unique(targets))
print("Number of unique words:", len(np.unique(np.hstack(data))))

#Categories: [0 1]
#Number of unique words: 9998
length = [len(i) for i in data]
print("Average Review length:", np.mean(length))
print("Standard Deviation:", round(np.std(length)))

#Average Review length: 234.75892
#Standard Deviation: 173.0

print('Maximum review length: {}'.format(max(length)))
print('Minimum review length: {}'.format(min(length)))

#Looking at a single training example:
print("Label:", targets[0])
#Label: 1
print(data[0])

index = imdb.get_word_index()
reverse_index = dict([(value, key) for (key, value) in index.items()]) 
decoded = " ".join( [reverse_index.get(i - 3, "#") for i in data[0]] )
print(decoded) 

#Creating Training(25,000)+Validation(15,000) and Test(10,000) sttartefied dataset - a split of 50-30-20% respectively
#First splitting dataset of 50,000 instances into training (80%) and test (20%)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data, targets,
                                                    stratify=targets, 
                                                    test_size=0.20)

#Further splitting trainng data such that final split represents 50% training (25000 instances), 30% validation (15000 instances)
X_training, X_validation, Y_training, Y_validation = train_test_split(X_train, Y_train,
                                                stratify=Y_train, #shuffle=True,
                                                test_size=0.375)
print("Size of Training dataset (50%):", len(X_training))
print("Size of Validation dataset (30%):", len(X_validation))
print("Size of Test dataset (20%):", len(X_test))

#Creating separate set of test set for testing Pre-trained word embeddings
X_test_embd, Y_test_embd = X_test, Y_test
X_training_embd, X_validation_embd, Y_training_embd, Y_validation_embd = X_training, X_validation, Y_training, Y_validation 


# In[ ]:


#Padding the token sequences to make all of them size
from tensorflow.keras.preprocessing import sequence
max_words = 500
X_training = sequence.pad_sequences(X_training, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
X_validation = sequence.pad_sequences(X_validation, maxlen=max_words)


# ## LSTM using Word Embedings learned On-the-fly

# In[ ]:


#Creating Training(25,000)+Validation(15,000) and Test(10,000) sttartefied dataset - a split of 50-30-20% respectively
#First splitting dataset of 50,000 instances into training (80%) and test (20%)
from sklearn.model_selection import train_test_split
X_train_emb_lrn_lstm, X_test_emb_lrn_lstm, Y_train_emb_lrn_lstm, Y_test_emb_lrn_lstm = train_test_split(data, targets,
                                                                                  stratify=targets, 
                                                                                  test_size=0.20)
print("Shape of newly created train and test segments respectively:",X_train_emb_lrn_lstm.shape, X_test_emb_lrn_lstm.shape)

#Now i will split 80% training data obtained above into further training and validation datasets
#This is done in a way such that final distribution of 50000 instance IMDB dataset looks like 50% training (25000 instances), 30% validation (15000 instances) and remaining test
#Please note that _emb_lrn has been suffixed to all variables in this part for the clarity of code

#Variable for total number of train-validation model fitting
shuffle=5
#This list wil have 5 validation accuracies by the end of below for loop
vlscores_emb_lrn_lstm=[]
for i in range(shuffle):
    print(i)
    #Further splitting such that final split represents 50% training (25000 instances), 30% validation (15000 instances)
    X_training_emb_lrn_lstm, X_validation_emb_lrn_lstm, Y_training_emb_lrn_lstm, Y_validation_emb_lrn_lstm = train_test_split(X_train_emb_lrn_lstm, Y_train_emb_lrn_lstm,
                                                                                                                              stratify=Y_train, shuffle=True,
                                                                                                                              test_size=0.375)
    print("Size of Training dataset (50%):", len(X_training_emb_lrn_lstm))
    print("Size of Validation dataset (30%):", len(X_validation_emb_lrn_lstm))
    
    #Padding the token sequences to make all of them size
    max_words = 500
    X_training_emb_lrn_lstm = sequence.pad_sequences(X_training_emb_lrn_lstm, maxlen=max_words)
    X_validation_emb_lrn_lstm = sequence.pad_sequences(X_validation_emb_lrn_lstm, maxlen=max_words)
    
    #LSTM
    #Building and Compiling the model over training data
    embedding_size=32
    model_lstm=Sequential()
    model_lstm.add(Embedding(10000, embedding_size, input_length=max_words)) #vocabulary_size
    model_lstm.add(tf.keras.layers.Dropout(0.2))
    model_lstm.add(LSTM(100))
    model_lstm.add(tf.keras.layers.Dense(256, activation='relu'))
    model_lstm.add(tf.keras.layers.Dropout(0.5))
    #Densely connected output layer with sigmoid as Activation function
    model_lstm.add(Dense(1, activation='sigmoid'))
    print(model_lstm.summary())
    #Compile model_lstm: Using ADAM optimiser and Binary Cross Entropy as it is a binary classification loss function 
    model_lstm.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

    batch_size = 512
    num_epochs = 100
    #Fitting the model_lstm on training data and validating it simultaneously
    hist = model_lstm.fit(X_training_emb_lrn_lstm, Y_training_emb_lrn_lstm, validation_data=(X_validation_emb_lrn_lstm, Y_validation_emb_lrn_lstm), batch_size=batch_size, epochs=num_epochs,verbose=1,callbacks=[es])

    #Calculating Accuracy metrics over Validation set 
    scores_emb_lrn_lstm = model_lstm.evaluate(X_validation_emb_lrn_lstm, Y_validation_emb_lrn_lstm, verbose=2)
    print("Validation set accuracy: %s: %.2f%%" % (model_lstm.metrics_names[1], scores_emb_lrn_lstm[1]*100))

    #Appending validation accuracy to a list 
    vlscores_emb_lrn_lstm.append(scores_emb_lrn_lstm[1] * 100)

print("Mean Accuracy score on Validation dataset: %.2f%% (+/- %.2f%%)" % (numpy.mean(vlscores_emb_lrn_lstm), numpy.std(vlscores_emb_lrn_lstm)))

# Visualize Costs over Epochs for last Validation (Doing this only once just to show a general trend)
plt.plot(hist.history['loss'],label="training loss")
plt.plot(hist.history['val_loss'],label="validation loss")
plt.ylabel("Cost")
plt.xlabel("Epoch")
plt.show()

#Testing the last model_lstm over our hold-out 20% Test dataset
X_test_emb_lrn_lstm = sequence.pad_sequences(X_test_emb_lrn_lstm, maxlen=max_words)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_emb_lrn_lstm, Y_test_emb_lrn_lstm))
print("Accuracy score on our Hold-out 20% Test dataset:" )
results = model_lstm.evaluate(test_dataset.batch(512), verbose=2)

