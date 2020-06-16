# # ** Working with your own Data **
# ### **Data Collection**
# 
# I have constructed a labelled dataset in form of a .csv file. The dataset has three variables - Review, Sentiment, and Movie. This dataset is freshly created from the actual IMDB movie database from https://www.imdb.com/search/title/ by taking 30 moview from my birth year 1987. For each movie, i have selected one 'Positive' and one 'Negative' review and all the reviews are in English. Also note that, I have included the titles of the reviews as well in the review text.I am supplying this raw data of 60 movie reviews along with my source code. 
# 

# ### **Modelling** 
# 
# *   Splitting new 30 movie review data into  70/30 between training and validation.
# *   Using my best performing model from Part 1,i.e. the Single Layer LSTM model trained on word embeddings learned on-the-fly, build a model on this small dataset and test the performance.
# *   Reporting Training and Validation scores for this fine-turned model and Saving this model too. I will be supplying this saved model along with the original modelas part of my assignment submissions. 
# *  Lastly, building a “from scratch” model for this novel smaller dataset of 30 movie reviews. I am using the exact same architecture of - single layer LSTM with embeddings learned on-the-fly and similar layers. 
# *  Comparing the performance of this “from scratch” model to the fine-tuned pre-trained model from Part1.
# 
# 
# 
# 
# 
# 

# ### **Preparing "From Scratch" model on 30 Movie database using exactly same architecture as my best saved model**
# Building a “from scratch” model for this novel smaller dataset of 30 movie reviews. I am using the exact same architecture of - single layer LSTM with embeddings learned on-the-fly and similar layers. The accuracy is coming very low as the vocab size is huge.

# This model will serve as a **baseline model** of performance, when i will investigate how the addition of transfer learning affects the performance on this problem. Please see below.

# In[ ]:


import pandas as pd
import io
from google.colab import files

uploaded = files.upload()
data_30 = pd.read_csv(io.BytesIO(uploaded['imdb_30_1987_1.csv'] )) #.decode('utf-8')))
# Dataset is now stored in a Pandas Dataframe

#First look at the imported data
print(data_30[0:5])

data_30 = data_30[["Review","Sentiment"]]
#Changing Positive to 1 and Negative to 0 for ease of processing
data_30.loc[data_30["Sentiment"] == "Positive", "Sentiment"] = 1
data_30.loc[data_30["Sentiment"] == "Negative", "Sentiment"] = 0
#Converting the datatype of sentiment variable in dataframe
data_30[['Sentiment']] = data_30[['Sentiment']].apply(pd.to_numeric) 
#Displaying datatypes of dataframe and first 5 rows
data_30.dtypes, data_30.head()
total_reviews = data_30.Review.values

#Data cleaning: Removing stopwords
#Making stopwords a 'set' instead of a 'list' as it is O(1) for sets while O(N) for lists
#from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stopwords_set = set(stopwords.words('english'))
data_30.Review = data_30.Review.str.lower()
data_30.Review = data_30.Review.apply(lambda x: [item for item in x.split() if item not in stopwords_set])
data_30.Review = data_30.Review.apply(lambda x: " ".join(x) )

#Setting up embedding dimension or output dim
embedding_size=32
#Maximum length allowed for a review 
max_words=500

#Variable for total number of train-validation model fitting
shuffle=5
#This list wil have 5 validation accuracies by the end of below for loop
vlscores_emb_30=[]

for i in range(shuffle):
    print(i)
    #Splitting into 70% training data and 30% as validation data
    from sklearn.model_selection import train_test_split
    X_training_emb_30, X_validation_emb_30, Y_training_emb_30, Y_validation_emb_30 = train_test_split(data_30.Review, data_30.Sentiment,
                                                                                                      stratify=data_30.Sentiment, #shuffle=True,
                                                                                                      test_size=0.3)

    print("Size of Training dataset (70%):", len(X_training_emb_30))
    print("Size of Validation dataset (30%):", len(X_validation_emb_30))

    #Converting pandas series to lists 
    X_training_emb_30 = X_training_emb_30.tolist()
    X_validation_emb_30 = X_validation_emb_30.tolist()
    Y_training_emb_30 = Y_training_emb_30.tolist()
    Y_validation_emb_30 = Y_validation_emb_30.tolist()

    # Make a tokenizer
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_training_emb_30)
    tokenizer.fit_on_texts(X_validation_emb_30)

    # Padding the tokens
    X_training_emb_30 = pad_sequences(tokenizer.texts_to_sequences(X_training_emb_30), maxlen=max_words)
    X_validation_emb_30 = pad_sequences(tokenizer.texts_to_sequences(X_validation_emb_30), maxlen=max_words)

    Y_validation_emb_30 = np.array(Y_validation_emb_30)
    Y_training_emb_30 = np.array(Y_training_emb_30)

    #LSTM
    #Building and Compiling the model over training data
    inp = Input(shape=(max_words,))
    emb = Embedding(10000, embedding_size)(inp)
    drop1 = Dropout(0.2)(emb)
    lstm = LSTM(100)(drop1) 
    dense = Dense(256, activation='relu')(lstm)
    drop2 = Dropout(0.5)(dense)
    #Densely connected output layer with sigmoid as Activation function
    out = Dense(1, activation='sigmoid')(drop2)
    model = Model(inputs=inp, outputs=out)
    #Compile model: Using ADAM optimiser and Binary Cross Entropy as it is a binary classification loss function 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    batchSize = 512
    num_epochs = 100
    #Fitting the model on training data and validating it simultaneously
    hist = model.fit(X_training_emb_30, Y_training_emb_30, validation_data=(X_validation_emb_30, Y_validation_emb_30),
                     batch_size=batchSize, epochs=num_epochs, verbose=1,callbacks=[es] )
    #hist = model.fit(x_tokenized, y, batch_size=batchSize, epochs=epochs, verbose=1, shuffle=True, validation_split=0.5)

    #Calculating Accuracy metrics over Validation set 
    scores_emb_30 = model.evaluate(X_validation_emb_30, Y_validation_emb_30, verbose=2)
    print("Validation set accuracy: %s: %.2f%%" % (model.metrics_names[1], scores_emb_30[1]*100))

    #Appending validation accuracy to a list 
    vlscores_emb_30.append(scores_emb_30[1] * 100)


# Visualize Costs over Epochs for last Validation (Doing this only once just to show a general trend)
plt.plot(hist.history['loss'],label="training loss")
plt.plot(hist.history['val_loss'],label="validation loss")
plt.ylabel("Cost")
plt.xlabel("Epoch")
plt.show()

#There is no Testing dataset in this case as the dataset was vey small (only 60 rows)
#Getting average accuracy over validation set from our list
print("Mean Accuracy score and standard devation on Validation dataset: %.2f%% (+/- %.2f%%)" % (numpy.mean(vlscores_emb_30), numpy.std(vlscores_emb_30)))


# #### **Re-Running the "from scratch" model with lower vocab size, lower maximum words per sentence.**
# Even though the question has instructions that we have to use the exact architecture from our previous best saved model, I am trying this run to check if we can train a good accuracy model from scratch by reducing vocab_size, maximum word length for a review and smaller batch size. All in order to accommodate the small size of dataset (60 instances only).
# 
# This too has resulted in a very poor performance, even after stratifying the training and validation split and doing 5-fold cross-validation on 70/30 training and validation split.

# In[ ]:


import pandas as pd
import io
from google.colab import files

uploaded = files.upload()
data_30 = pd.read_csv(io.BytesIO(uploaded['imdb_30_1987_1.csv'] )) #.decode('utf-8')))
# Dataset is now stored in a Pandas Dataframe

#First look at the imported data
print(data_30[0:5])

data_30 = data_30[["Review","Sentiment"]]
#Changing Positive to 1 and Negative to 0 for ease of processing
data_30.loc[data_30["Sentiment"] == "Positive", "Sentiment"] = 1
data_30.loc[data_30["Sentiment"] == "Negative", "Sentiment"] = 0
#Converting the datatype of sentiment variable in dataframe
data_30[['Sentiment']] = data_30[['Sentiment']].apply(pd.to_numeric) 
#Displaying datatypes of dataframe and first 5 rows
data_30.dtypes, data_30.head()
total_reviews = data_30.Review.values

#Data cleaning: Removing stopwords
#Making stopwords a 'set' instead of a 'list' as it is O(1) for sets while O(N) for lists
#from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stopwords_set = set(stopwords.words('english'))
data_30.Review = data_30.Review.str.lower()
data_30.Review = data_30.Review.apply(lambda x: [item for item in x.split() if item not in stopwords_set])
data_30.Review = data_30.Review.apply(lambda x: " ".join(x) )

#Setting up embedding dimension or output dim
embedding_size=32
#Maximum length allowed for a review 
max_words=100

#Variable for total number of train-validation model fitting
shuffle=5
#This list wil have 5 validation accuracies by the end of below for loop
vlscores_emb_30_chng=[]

for i in range(shuffle):
    print(i)
    #Splitting into 70% training data and 30% as validation data
    from sklearn.model_selection import train_test_split
    X_training_emb_30, X_validation_emb_30, Y_training_emb_30, Y_validation_emb_30 = train_test_split(data_30.Review, data_30.Sentiment,
                                                                                                      stratify=data_30.Sentiment, #shuffle=True,
                                                                                                      test_size=0.3)

    print("Size of Training dataset (70%):", len(X_training_emb_30))
    print("Size of Validation dataset (30%):", len(X_validation_emb_30))

    #Converting pandas series to lists 
    X_training_emb_30 = X_training_emb_30.tolist()
    X_validation_emb_30 = X_validation_emb_30.tolist()
    Y_training_emb_30 = Y_training_emb_30.tolist()
    Y_validation_emb_30 = Y_validation_emb_30.tolist()

    # Make a tokenizer
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(data_30.Review.tolist())
    #tokenizer.fit_on_texts(X_validation_emb_30)

    # Padding the tokens
    X_training_emb_30 = pad_sequences(tokenizer.texts_to_sequences(X_training_emb_30), maxlen=max_words)
    X_validation_emb_30 = pad_sequences(tokenizer.texts_to_sequences(X_validation_emb_30), maxlen=max_words)

    Y_validation_emb_30 = np.array(Y_validation_emb_30)
    Y_training_emb_30 = np.array(Y_training_emb_30)

    #LSTM
    #Building and Compiling the model over training data
    inp = Input(shape=(max_words,))
    emb = Embedding(1000, embedding_size)(inp)
    drop1 = Dropout(0.2)(emb)
    lstm = LSTM(100)(drop1) 
    dense = Dense(256, activation='relu')(lstm)
    drop2 = Dropout(0.5)(dense)
    #Densely connected output layer with sigmoid as Activation function
    out = Dense(1, activation='sigmoid')(drop2)
    model = Model(inputs=inp, outputs=out)
    #Compile model: Using ADAM optimiser and Binary Cross Entropy as it is a binary classification loss function 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    batchSize = 32
    num_epochs = 100
    #Fitting the model on training data and validating it simultaneously
    hist = model.fit(X_training_emb_30, Y_training_emb_30, validation_data=(X_validation_emb_30, Y_validation_emb_30),
                     batch_size=batchSize, epochs=num_epochs, verbose=1,callbacks=[es] )
    #hist = model.fit(x_tokenized, y, batch_size=batchSize, epochs=epochs, verbose=1, shuffle=True, validation_split=0.5)

    #Calculating Accuracy metrics over Validation set 
    scores_emb_30 = model.evaluate(X_validation_emb_30, Y_validation_emb_30, verbose=2)
    print("Validation set accuracy: %s: %.2f%%" % (model.metrics_names[1], scores_emb_30[1]*100))

    #Appending validation accuracy to a list 
    vlscores_emb_30_chng.append(scores_emb_30[1] * 100)


# Visualize Costs over Epochs for last Validation (Doing this only once just to show a general trend)
plt.plot(hist.history['loss'],label="training loss")
plt.plot(hist.history['val_loss'],label="validation loss")
plt.ylabel("Cost")
plt.xlabel("Epoch")
plt.show()

#There is no Testing dataset in this case as the dataset was vey small (only 60 rows)
#Getting average accuracy over validation set from our list
print("Accuracy score on Validation dataset: %.2f%% (+/- %.2f%%)" % (numpy.mean(vlscores_emb_30), numpy.std(vlscores_emb_30)))


# ## **Transfer Learning**
# The model_lstm that was fit on large movie review database, can be loaded and its weights can be used as the initial weights for a model fit on smaller movie review database classification problem.
# 
# In this type of **Transfer learning**, we implement learning on a different but related problem set and the model weighgts are used for weight initialization.
# 
# For this we need to reload the saved model from LSTM problem and refit it on examples of Smaller movie review dataset. Once loaded, the model can be compiled and fit as per normal.

# My main model or best model has five layers including the input embedding layer - 
# Embedding layer,
# Dropout layer on inputs,
# LSTM layer for context learning,
# Dense RELU layer, and 
# Dropout layer for hidden layer.
# 
# There are **TWO** types of Transfer Learninig methodologies:
# * **Weight Initialization** - if all the network weights are allowed to change or adapt to the new dataset, then transfer learning simply serves as a weight initialization method.
# 
# * **Feature Extraction Method** - if only weigths in output layer is allowed to change while all hidden layer's weights are kept fixed, then transfer learning serves as a feature extraction method.
# 
# 
# 

# ### **Model Comparisons**
# Now, if we will let all the network weights to be adapted, then its probably not a good idea. This is because the target dataset is very small (60 rows only). Thus if we will allow only some weights in the network to adapt or if we can only let the output layer train (act as a feature extractor), we should get better results. Let's examine tis through our code -
# Please not that here I am doing a stratified split of training and validation into 70/30 sets and doing 5-fold cross validation after freezing 0,1,2,3,4 and all 5 rows individually. I am then plotting training and validation loss after freezing each row one by one. More will be clear by looking at the code and its output. 

# In[ ]:


#Creating a list to append mean validation accuracies after freezing 0,1,2,3,4 and all 5 rows consequtively. 
freeze_scores =[]
#Total number of layers in the saved or loaded model
fixed_layers=5
#Running loop to see the effect on freezing 1,2,3,4 or all 5 rows from starting of the network
#Checking this effect for 5 cross validation datasets to make our analysis more robust
for lyrs in range(fixed_layers):
    # load a saved model
    model_transfer = load_model('/content/gdrive/My Drive/Colab Notebooks/model_lstm_d18129636.h5')

    #Freezing the number of layers. 
    #layers[0] would mean no layer is on freeze, all network weights can adapt
    #layers[5] would mean all layers are on freeze 
    model_transfer.layers[lyrs].trainable = False
    # re-compile the loaded model
    model_transfer.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    shuffle=5
    #This list wil have 5 validation accuracies by the end of below for loop
    vlscores=[]

    for i in range(shuffle):
        print(i)
        #Splitting into 70% training data and 30% as validation data
        from sklearn.model_selection import train_test_split
        X_training_emb_30, X_validation_emb_30, Y_training_emb_30, Y_validation_emb_30 = train_test_split(data_30.Review, data_30.Sentiment,
                                                                                                          stratify=data_30.Sentiment, #shuffle=True,
                                                                                                          test_size=0.3)

        #Converting pandas series to lists 
        X_training_emb_30 = X_training_emb_30.tolist()
        X_validation_emb_30 = X_validation_emb_30.tolist()
        Y_training_emb_30 = Y_training_emb_30.tolist()
        Y_validation_emb_30 = Y_validation_emb_30.tolist()

        # Make a tokenizer
        tokenizer = Tokenizer(num_words=1000)
        tokenizer.fit_on_texts(data_30.Review.tolist())
        #tokenizer.fit_on_texts(X_validation_emb_30)

        # Padding the tokens
        X_training_emb_30 = pad_sequences(tokenizer.texts_to_sequences(X_training_emb_30), maxlen=max_words)
        X_validation_emb_30 = pad_sequences(tokenizer.texts_to_sequences(X_validation_emb_30), maxlen=max_words)

        Y_validation_emb_30 = np.array(Y_validation_emb_30)
        Y_training_emb_30 = np.array(Y_training_emb_30)
        num_epochs = 20

        # fit loaded model on new training dataset and validate it
        history = model_transfer.fit(X_training_emb_30, Y_training_emb_30, validation_data=(X_validation_emb_30, Y_validation_emb_30), epochs=num_epochs, verbose=1)

        #Calculating Accuracy metrics over Validation set 
        scores = model_transfer.evaluate(X_validation_emb_30, Y_validation_emb_30, verbose=2)
        print("Validation set accuracy: %s: %.2f%%" % (model_transfer.metrics_names[1], scores[1]*100))

        #Appending validation accuracy to a list 
        vlscores.append(scores[1] * 100)

    #Saving the best model
    if lyrs==3:
      #Using model.save() to save the weights of the model along with architecture in HDF5 format
      model_transfer.save("/content/gdrive/My Drive/Colab Notebooks/model_transfer_d18129636.h5")
      print("Saved second model to mounted google-drive")

    #Appending validation accuracy to a list 
    freeze_scores.append(vlscores)

    # Visualize Costs over Epochs for last Validation )
    print("Freezing first -",lyrs ,"layers and Plotting Loss for training and validation" )
    plt.plot(history.history['loss'],label="training loss")
    plt.plot(history.history['val_loss'],label="validation loss")
    plt.ylabel("Cost")
    plt.xlabel("Epoch")
    plt.show()


# In[ ]:


#Listing all 5 validation accuracies for evry freeze
print(freeze_scores)

#Getting average validation set accuracy for every freeze from our list
for i in range(len(freeze_scores)):
  print("Accuracy score on Validation dataset after freezing ",i, " rows : %.2f%% (+/- %.2f%%)" % (numpy.mean(freeze_scores[i]), numpy.std(freeze_scores[i])))
print("scores in - Mean +/- standard deviations")


# 
# Mean Accuracy score and standard devation on Validation dataset for ouurbaseline model was: 60.00% (+/- 8.89%)
# 

# In[ ]:


#Getting average accuracy over validation set from our list - for our baseline "from scratch" model (code is present above)
print("Mean Accuracy score and standard devation on Validation dataset for our baseline 'from scratch' model: %.2f%% (+/- %.2f%%)" % (numpy.mean(vlscores_emb_30), numpy.std(vlscores_emb_30)))

#Mean Accuracy score and standard devation on Validation dataset: 60.00% (+/- 8.89%)
#vlscores_emb_30


# #### Comparing Mean Validation set Accuracies for all models
# Standalone | 60.00% (+/- 8.89%)
# --- | ---
# Transfer (fixed=0 rows) | 70.00% (+/- 7.54%)
# Transfer (fixed=1 rows) | 85.56% (+/- 17.07%)
# Transfer (fixed=2 rows) | 88.89% (+/- 19.56%)
# Transfer (fixed=3 rows) | 88.89% (+/- 9.94%)
# Transfer (fixed=4 rows) | 86.67% (+/- 13.88%)
# 

# **As expected in transfer learning, the training took very less time in terms of the learning curve in all the cases as shown above. Our new model used the weights from older saved best model, fitted on a larger dataset from a related problem. long with learning speed, this transfer learning also resulted in lower generalization error as is evideny from the validation scores.**
# 
# ### Lastly, the best model obtained by applying transfer learning is the one in which first 3 rows were frozen and rest served in the weight initialization scheme. 
# ### This model has best accuracy and minimum standard deviation - 88.89% +/- 9.94%
# ### I have already Saved the model thus learned in above steps

# In[ ]:



