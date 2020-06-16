
# ##**Embeddings**
# ## Sentiment analysis using Pre-trained word embeddings and Dense RELU
# Getting the dataset in form of csv from Kaggle using Kaggle API. In this step the reason I am using using data downloaded from kaggle and not the one available from Keras is that I need to pass string tensors as an input to the pre-trained word embedding that I am using from the Tensorflow Hub. The IMDB dataset available on Keras is a sequence of numbers and not words and hence not using it in this specific model. Please note that the two datasets are completely identical. 

# In[ ]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import numpy
import pandas as pd
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import nltk
nltk.download('stopwords')
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

#Employing Early stopping as a means of Validation
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

#Deciding on number of epochs
#Number of Epochs is how many time the entire dataset has to pass through the network
num_epochs = 100
#Using pre-trained embedding
#embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
#embedding = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1"
embedding = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], #output_shape=[128], 
                           dtype=tf.string, trainable=False)


#Getting my IMDB data from Kaggle
import kaggle
kaggle.api.authenticate()
#Specifying the required IMDB database name to kaggle API
kaggle.api.dataset_download_files('lakshmi25npathi/imdb-dataset-of-50k-movie-reviews', path='./tmp', unzip=True)
data_kaggle = pd.read_csv('./tmp/IMDB Dataset.csv')
data_kaggle.head()

#data = data[:20000]
#Changing Positive to 1 and Negative to 0 for ease of processing
data_kaggle.loc[data_kaggle["sentiment"] == "positive", "sentiment"] = 1
data_kaggle.loc[data_kaggle["sentiment"] == "negative", "sentiment"] = 0
#Converting the datatype of sentiment variable in dataframe
data_kaggle[['sentiment']] = data_kaggle[['sentiment']].apply(pd.to_numeric) 
#Displaying datatypes of dataframe and first 5 rows
data_kaggle.dtypes, data_kaggle.head()
total_reviews = data_kaggle.review.values

#Data cleaning: Removing stopwords
#Making stopwords a 'set' instead of a 'list' as it is O(1) for sets while O(N) for lists
#from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stopwords_set = set(stopwords.words('english'))
data_kaggle.review = data_kaggle.review.str.lower()
data_kaggle.review = data_kaggle.review.apply(lambda x: [item for item in x.split() if item not in stopwords_set])
data_kaggle.review = data_kaggle.review.apply(lambda x: " ".join(x) )


#Creating Training(25,000)+Validation(15,000) and Test(10,000) sttartefied dataset - a split of 50-30-20% respectively
#First splitting dataset of 50,000 instances into training (80%) and test (20%)
from sklearn.model_selection import train_test_split
X_train_emb_pretrnd, X_test_emb_pretrnd, Y_train_emb_pretrnd, Y_test_emb_pretrnd = train_test_split(data_kaggle.review, data_kaggle.sentiment,
                                                                                  stratify=data_kaggle.sentiment,#shuffle=True,
                                                                                  test_size=0.20)
print("Shape of newly created train and test segments respectively:",X_train_emb_pretrnd.shape, X_test_emb_pretrnd.shape)

#Creating inputs in the format that is required for the hub _layer (a 1D tensor of strings)
test_dataset_pre = tf.data.Dataset.from_tensor_slices((X_test_emb_pretrnd, Y_test_emb_pretrnd))

#Now i will split 80% training data obtained above into further training and validation datasets
#This is done in a way such that final distribution of 50000 instance IMDB dataset looks like 50% training (25000 instances), 30% validation (15000 instances) and remaining test
#Please note that _emb_lrn has been suffixed to all variables in this part for the clarity of code

#Variable for total number of train-validation model fitting
shuffle=5
#This list wil have 5 validation accuracies by the end of below for loop
vlscores_emb_pretrnd=[]
for i in range(shuffle):
    print(i)
    #Further splitting such that final split represents 50% training (25000 instances), 30% validation (15000 instances)
    X_training_emb_pretrnd, X_validation_emb_pretrnd, Y_training_emb_pretrnd, Y_validation_emb_pretrnd = train_test_split(X_train_emb_pretrnd, Y_train_emb_pretrnd,
                                                                                                                              stratify=Y_train_emb_pretrnd, #shuffle=True,
                                                                                                                              test_size=0.375)
    print("Size of Training dataset (50%):", len(X_training_emb_pretrnd))
    print("Size of Validation dataset (30%):", len(X_validation_emb_pretrnd))
    
    #Padding the token sequences to make all of them size
    #max_words = 500
    #X_training_emb_pretrnd = sequence.pad_sequences(X_training_emb_pretrnd, maxlen=max_words)
    #X_validation_emb_pretrnd = sequence.pad_sequences(X_validation_emb_pretrnd, maxlen=max_words)
    
    #Creating inputs in the format that is required for the hub _layer (a 1D tensor of strings)
    train_dataset_pre = tf.data.Dataset.from_tensor_slices((X_training_emb_pretrnd, Y_training_emb_pretrnd))
    val_dataset_pre = tf.data.Dataset.from_tensor_slices((X_validation_emb_pretrnd, Y_validation_emb_pretrnd))

    #Pre-trained word embedding from Hub
    #Building and Compiling the model
    model=tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())

    model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

    #Setting up batch size as 512 and fitting the model by training on 50% data while validating on 30% of data 
    batch_size = 512
    num_epochs = 100
    hist = model.fit(train_dataset_pre.shuffle(10000).batch(batch_size),
                    epochs=num_epochs,
                    validation_data=val_dataset_pre.batch(batch_size),
                    verbose=1,
                    callbacks=[es])

    #Calculating Accuracy metrics over Validation set 
    scores_emb_pretrnd = model.evaluate(X_validation_emb_pretrnd, Y_validation_emb_pretrnd, verbose=2)
    print("Validation set accuracy: %s: %.2f%%" % (model.metrics_names[1], scores_emb_pretrnd[1]*100))

    #Appending validation accuracy to a list 
    vlscores_emb_pretrnd.append(scores_emb_pretrnd[1] * 100)


# Visualize Costs over Epochs for last Validation (Doing this only once just to show a general trend)
plt.plot(hist.history['loss'],label="training loss")
plt.plot(hist.history['val_loss'],label="validation loss")
plt.ylabel("Cost")
plt.xlabel("Epoch")
plt.show()

#Getting average accuracy over validation set from our list
print("Accuracy score on Validation dataset: %.2f%% (+/- %.2f%%)" % (numpy.mean(vlscores_emb_pretrnd), numpy.std(vlscores_emb_pretrnd)))

#Testing the last model over our hold-ut 20% Test dataset
#test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
print("Testing the last model over our hold-out 20% Test dataset")
print("Loss and Accuracy Test dataset :-")
results = model.evaluate(test_dataset_pre.batch(512), verbose=2)

#Stating Accuracy over hold-out 20% test set
print("Loss and Accuracy Test dataset :-")
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))

