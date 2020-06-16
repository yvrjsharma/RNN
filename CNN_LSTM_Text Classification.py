
# ## **CNN+LSTM for Text Classification**
# Investigating the use of CNNs with **Multiple Filters** as an additional layer before a **LSTM** solution. 
# 

# In[85]:


import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Embedding, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.datasets import imdb
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,LSTM, Embedding, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling3D, MaxPooling1D

max_features = 5000
maxlen = 400
embedding_dims = 50
embedding_size =32

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)



model = Sequential()
model.add(Embedding(max_features,
                    embedding_size,
                    input_length=maxlen))
model.add(Conv1D(200,3,padding='valid',activation='relu',strides=1))
#model.add(GlobalMaxPooling1D())
model.add(MaxPooling1D(pool_size=3))
model.add(Dense(200, activation='relu'))
model.add(LSTM(100))
model.add(Dense(200, activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

hist = model.fit(x_train, y_train,
          batch_size=32,
          epochs=10,
          validation_data=(x_test, y_test))


# Visualize Costs over Epochs for last Validation (Doing this only once just to show a general trend)
plt.plot(hist.history['loss'],label="training loss")
plt.plot(hist.history['val_loss'],label="validation loss")
plt.ylabel("Cost")
plt.xlabel("Epoch")
plt.show()


