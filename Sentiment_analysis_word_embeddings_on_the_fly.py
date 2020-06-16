
# ## Sentiment analysis using word embeddings learned on the fly and Dense RELU

# In[ ]:


#Creating Training(25,000)+Validation(15,000) and Test(10,000) sttartefied dataset - a split of 50-30-20% respectively
#First splitting dataset of 50,000 instances into training (80%) and test (20%)
from sklearn.model_selection import train_test_split
X_train_emb_lrn, X_test_emb_lrn, Y_train_emb_lrn, Y_test_emb_lrn = train_test_split(data, targets,
                                                                                  stratify=targets, 
                                                                                  test_size=0.20)
print("Shape of newly created train and test segments respectively:",X_train_emb_lrn.shape, X_test_emb_lrn.shape)

#Now i will split 80% training data obtained above into further training and validation datasets
#This is done in a way such that final distribution of 50000 instance IMDB dataset looks like 50% training (25000 instances), 30% validation (15000 instances) and remaining test
#Please note that _emb_lrn has been suffixed to all variables in this part for the clarity of code

#Variable for total number of train-validation model fitting
shuffle=5
#This list wil have 5 validation accuracies by the end of below for loop
vlscores_emb_lrn=[]
for i in range(shuffle):
    print(i)
    #Further splitting such that final split represents 50% training (25000 instances), 30% validation (15000 instances)
    X_training_emb_lrn, X_validation_emb_lrn, Y_training_emb_lrn, Y_validation_emb_lrn = train_test_split(X_train_emb_lrn, Y_train_emb_lrn,
                                                    stratify=Y_train_emb_lrn, shuffle=True,
                                                    test_size=0.375)
    print("Size of Training dataset (50%):", len(X_training_emb_lrn))
    print("Size of Validation dataset (30%):", len(X_validation_emb_lrn))
    
    #Padding the token sequences to make all of them size
    max_words = 500
    X_training_emb_lrn = sequence.pad_sequences(X_training_emb_lrn, maxlen=max_words)
    X_validation_emb_lrn = sequence.pad_sequences(X_validation_emb_lrn, maxlen=max_words)
    
    #Building and Compiling the model over training data
    embedding_size=32
    model=tf.keras.Sequential()
    model.add(Embedding(10000, embedding_size, input_length=max_words)) #vocabulary_size
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    #Densely connected output layer with sigmoid as Activation function
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    #Compile model: Using ADAM optimiser and Binary Cross Entropy as it is a binary classification loss function 
    model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])


    batch_size = 512
    num_epochs = 100
    #Fitting the model on training data and validating it simultaneously
    hist = model.fit(X_training_emb_lrn, Y_training_emb_lrn, validation_data=(X_validation_emb_lrn, Y_validation_emb_lrn), batch_size=batch_size, epochs=num_epochs,verbose=1,callbacks=[es])

    #Calculating Accuracy metrics over Validation set 
    scores_emb_lrn = model.evaluate(X_validation_emb_lrn, Y_validation_emb_lrn, verbose=2)
    print("Validation set accuracy: %s: %.2f%%" % (model.metrics_names[1], scores_emb_lrn[1]*100))

    #Appending validation accuracy to a list 
    vlscores_emb_lrn.append(scores_emb_lrn[1] * 100)


# Visualize Costs over Epochs for last Validation (Doing this only once just to show a general trend)
plt.plot(hist.history['loss'],label="training loss")
plt.plot(hist.history['val_loss'],label="validation loss")
plt.ylabel("Cost")
plt.xlabel("Epoch")
plt.show()

#Getting average accuracy over validation set from our list
print("Accuracy score on Validation dataset: %.2f%% (+/- %.2f%%)" % (numpy.mean(vlscores_emb_lrn), numpy.std(vlscores_emb_lrn)))

#Testing the last model over our hold-ut 20% Test dataset
#Converting numpy array to a tensor with paddings
X_test_emb_lrn = sequence.pad_sequences(X_test_emb_lrn, maxlen=max_words)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_emb_lrn, Y_test_emb_lrn))
print("Testing the last model over our hold-out 20% Test dataset")
print("Loss and Accuracy Test dataset :-")
results = model.evaluate(test_dataset.batch(512), verbose=2)

#Stating Accuracy over hold-out 20% test set
print("Loss and Accuracy Test dataset :-")
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))

