
# ## **CNN for Text Classification**
# Investigating the use of CNNs with **Multiple** and **Heterogeneous Kernel** sizes as an alternative to an LSTM solution.

# In[ ]:


#Creating Training(25,000)+Validation(15,000) and Test(10,000) sttartefied dataset - a split of 50-30-20% respectively
#First splitting dataset of 50,000 instances into training (80%) and test (20%)
from sklearn.model_selection import train_test_split
X_train_emb_lrn_cnn, X_test_emb_lrn_cnn, Y_train_emb_lrn_cnn, Y_test_emb_lrn_cnn = train_test_split(data, targets,
                                                                                  stratify=targets, 
                                                                                  test_size=0.20)
print("Shape of newly created train and test segments respectively:",X_train_emb_lrn_cnn.shape, X_test_emb_lrn_cnn.shape)

#Now i will split 80% training data obtained above into further training and validation datasets
#This is done in a way such that final distribution of 50000 instance IMDB dataset looks like 50% training (25000 instances), 30% validation (15000 instances) and remaining test
#Please note that _emb_lrn has been suffixed to all variables in this part for the clarity of code

#Variable for total number of train-validation model fitting
shuffle=5
#This list wil have 5 validation accuracies by the end of below for loop
vlscores_emb_lrn_cnn=[]
for i in range(shuffle):
    print(i)
    #Further splitting such that final split represents 50% training (25000 instances), 30% validation (15000 instances)
    X_training_emb_lrn_cnn, X_validation_emb_lrn_cnn, Y_training_emb_lrn_cnn, Y_validation_emb_lrn_cnn = train_test_split(X_train_emb_lrn_cnn, Y_train_emb_lrn_cnn,
                                                    stratify=Y_train_emb_lrn_cnn, shuffle=True,
                                                    test_size=0.375)
    print("Size of Training dataset (50%):", len(X_training_emb_lrn_cnn))
    print("Size of Validation dataset (30%):", len(X_validation_emb_lrn_cnn))
    
    #Padding the token sequences to make all of them same size
    max_words = 500
    X_training_emb_lrn_cnn = sequence.pad_sequences(X_training_emb_lrn_cnn, maxlen=max_words)
    X_validation_emb_lrn_cnn = sequence.pad_sequences(X_validation_emb_lrn_cnn, maxlen=max_words)
    
    #Creating three inputs for the three filter layers
    input1 = Input(500,) #max_words
    input2 = Input(500,)
    input3 = Input(500,)

    #CNN
    #Creating Multiple(more than one; 250 in this case) and Heterogeneous (means different kernel size OF 3,4, AND 5) filters
    #Creating first filter layer with Kernel size=3
    submodel1 = Sequential()
    submodel1.add(Embedding(10000,embedding_size,input_length=max_words))
    submodel1.add(Conv1D(250,3,padding='valid',activation='relu',strides=1))
    submodel1.add(GlobalMaxPooling1D())
    submodel1=submodel1(input1)

    #Creating second filter layer with Kernel size=4
    submodel2 = Sequential()
    submodel2.add(Embedding(10000,embedding_size,input_length=max_words))
    submodel2.add(Conv1D(250,4,padding='valid',activation='relu',strides=1))
    submodel2.add(GlobalMaxPooling1D())
    submodel2=submodel2(input2)

    #Creating third filter layer with Kernel size=5
    submodel3 = Sequential()
    submodel3.add(Embedding(10000,embedding_size,input_length=max_words))
    submodel3.add(Conv1D(250,5,padding='valid',activation='relu',strides=1))
    submodel3.add(GlobalMaxPooling1D())
    submodel3=submodel3(input3)

    #Using just a Sequential model for creating branches of three different filters is not possible
    #I am using some functions from functional API here for it
    #Here we will create layers and call them by passing tensors to get outputs
    conc = Concatenate()([submodel1, submodel2, submodel3])
    out = Dense(200, activation='relu')(conc)
    out = Dense(1, activation='sigmoid')(out)

    #Using Model class from tensorflow.keras.models.Model
    #Model groups layers into an object with training and output features
    modelall = Model(inputs=[input1, input2, input3],outputs= out)
    #To see the architrecture of model thus created
    modelall.summary()
    #Compille the new model created
    modelall.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    
    batch_size = 512
    num_epochs = 100
    #Fitting on Training data while validating it on Validation set
    #This model requires stacked input
    hist = modelall.fit([X_training_emb_lrn_cnn, X_training_emb_lrn_cnn, X_training_emb_lrn_cnn],
                     Y_training_emb_lrn_cnn, batch_size=batch_size,epochs=num_epochs,verbose=1,callbacks=[es],
                     validation_data=([X_validation_emb_lrn_cnn, X_validation_emb_lrn_cnn, X_validation_emb_lrn_cnn], Y_validation_emb_lrn_cnn))

    #Calculating Accuracy metrics over Validation set 
    scores_emb_lrn_cnn = modelall.evaluate([X_validation_emb_lrn_cnn, X_validation_emb_lrn_cnn, X_validation_emb_lrn_cnn], Y_validation_emb_lrn_cnn, verbose=2)
    print("Validation set accuracy: %s: %.2f%%" % (modelall.metrics_names[1], scores_emb_lrn_cnn[1]*100))

    #Appending validation accuracy to a list 
    vlscores_emb_lrn_cnn.append(scores_emb_lrn_cnn[1] * 100)


# Visualize Costs over Epochs for last Validation (Doing this only once just to show a general trend)
plt.plot(hist.history['loss'],label="training loss")
plt.plot(hist.history['val_loss'],label="validation loss")
plt.ylabel("Cost")
plt.xlabel("Epoch")
plt.show()

#Getting average accuracy over validation set from our list
print("Accuracy score on Validation dataset: %.2f%% (+/- %.2f%%)" % (numpy.mean(vlscores_emb_lrn_cnn), numpy.std(vlscores_emb_lrn_cnn)))

#Testing the last model over our hold-ut 20% Test dataset
#Converting numpy array to a tensor with paddings
X_test_emb_lrn_cnn = sequence.pad_sequences(X_test_emb_lrn_cnn, maxlen=max_words)
print("Testing the last model over our hold-out 20% Test dataset")
print("Loss and Accuracy Test dataset :-")
results = modelall.evaluate([X_test_emb_lrn_cnn,X_test_emb_lrn_cnn,X_test_emb_lrn_cnn], Y_test_emb_lrn_cnn, verbose=2)

#Stating Accuracy over hold-out 20% test set
print("Loss and Accuracy Test dataset :-")
for name, value in zip(modelall.metrics_names, results):
  print("%s: %.3f" % (name, value))

