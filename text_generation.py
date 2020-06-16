
# # **Writing your own Reviews**

# ## Getting the Text data of IMDB Movie Reviews using Kaggle API
# ### Reviewing the three set of texts -
# ### 1. Positive Text
# ### 2. Negative Text
# ### 3. Negative-Positive Text combined 
# 
# 

# In[ ]:


#Importing relevant libraries
import os
import nltk
from nltk.util import ngrams
#from nltk.lm import MLE
from nltk import word_tokenize
# we need to download a special component that is used by the tokenizer below 
nltk.download('punkt')
import tensorflow as tf
import numpy as np
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


# In[ ]:


data_kaggle_pos = data_kaggle[data_kaggle["sentiment"]==1]
data_kaggle_neg = data_kaggle[data_kaggle["sentiment"]==0]
#Review one sample text
print(data_kaggle.review[0]) 


# In[ ]:


all_neg_reviews = []
all_pos_reviews = []
all_reviews =[]
counter1 =0
counter2 =0 
counter3 =0
for review in data_kaggle_neg.review:
  all_neg_reviews.append(review)
  counter1+=1
for review in data_kaggle_pos.review:
  all_pos_reviews.append(review)
  counter2+=1
for review in data_kaggle.review:
  all_reviews.append(review)
  counter3+=1

#Total rows traverse in every dataframe - negative and positive   
print("Total reviews added to negative and positve respectively:",counter1,counter2)
#Creating strings of negative and positive reviews from respective lists
all_neg_reviews = " ".join(all_neg_reviews)
all_pos_reviews = " ".join(all_pos_reviews)
all_reviews = " ".join(all_reviews)

#Reviweing Negative sample text
print("Negative Sample : ",all_neg_reviews[:1000])
#Reviweing Positive sample text
print("Positive Sample : ",all_pos_reviews[:1000])


# In[ ]:


#TAking subset of data as this is highly memory intensive analysis
extract = 0.2
all_neg_reviews = all_neg_reviews[:int(extract*len(all_neg_reviews))]
all_pos_reviews = all_pos_reviews[:int(extract*len(all_pos_reviews))]
all_reviews = all_reviews[:int(extract*len(all_reviews))]


# In[ ]:


#To save checkpoints or models
my_temp_folder = "/content/gdrive/My Drive/Colab Notebooks/"


# ### Preparing our model on Negative, Positive and ALL Review Texts 
# * Preparing vocabulary with indices
# * Setting up dataset
# * Setting up input and output data for training
# 

# In[ ]:


#Setting up vocabularies and datasets for Negative, Positive and ALL
textneg = all_neg_reviews
textpos = all_pos_reviews
textall = all_reviews

# extract an ordered vocabulary - this will be letters, some numbers etc. 
vocabneg = sorted(set(textneg))
vocabpos = sorted(set(textpos))
vocaball = sorted(set(textall))

# Create mappings between vocab and numeric indices
char2idxneg = {u:i for i, u in enumerate(vocabneg)}
idx2charneg = np.array(vocabneg)

char2idxpos = {u:i for i, u in enumerate(vocabpos)}
idx2charpos = np.array(vocabpos)

char2idxall = {u:i for i, u in enumerate(vocaball)}
idx2charall = np.array(vocaball)


# Map all the training text over to a numeric representation
text_as_intneg = np.array([char2idxneg[c] for c in textneg])
text_as_intpos = np.array([char2idxpos[c] for c in textpos])
text_as_intall = np.array([char2idxall[c] for c in textall])

# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epochneg = len(textneg)//(seq_length+1)
examples_per_epochpos = len(textpos)//(seq_length+1)
examples_per_epochall = len(textall)//(seq_length+1)

# Create training examples / targets
char_datasetneg = tf.data.Dataset.from_tensor_slices(text_as_intneg)
sequencesneg = char_datasetneg.batch(seq_length+1, drop_remainder=True)

char_datasetpos = tf.data.Dataset.from_tensor_slices(text_as_intpos)
sequencespos = char_datasetpos.batch(seq_length+1, drop_remainder=True)

char_datasetall = tf.data.Dataset.from_tensor_slices(text_as_intall)
sequencesall = char_datasetall.batch(seq_length+1, drop_remainder=True)

# This function would create input and output training data for our model on the fly
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


# Mapping the above function to entire dataset 
datasetneg = sequencesneg.map(split_input_target)
datasetpos = sequencespos.map(split_input_target)
datasetall = sequencesall.map(split_input_target)

#Setting up hyperparameters
embedding_dim = 256
BATCH_SIZE = 64
BUFFER_SIZE = 10000
rnn_units = 2048

#Tensorflow wil use this buffer size and batch size to work with the problem data
datasetneg = datasetneg.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
datasetpos = datasetpos.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
datasetall = datasetall.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


# ###  Designing the Neural Languagage Model 

# In[ ]:


#To generate some new text we first define a function that can be used to generate this text
#The function takes the trained neural language model as input and a chunk of text that is used to 'prime' or seed the generator
def generate_text(model, start_string, chr2,idx2):

    char2idx = chr2
    idx2char = idx2

    # Number of characters to generate
    num_generate = 100

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    text_ids_gen = []
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_ids_gen.append(predicted_id)
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


# In[ ]:




#Desiging our model - Negative
mneg = tf.keras.Sequential()
#An embedding layer to capture meaning of words in a real valued space.
mneg.add(tf.keras.layers.Embedding(len(vocabneg), embedding_dim,
                          batch_input_shape=[BATCH_SIZE, None]))
#Recurrent layer - Long Short Term Memory
mneg.add(tf.keras.layers.LSTM(rnn_units,
                    return_sequences=True,
                    stateful=True,
                    recurrent_initializer='glorot_uniform'))
#Final output layer with one neuron per vovab entry
mneg.add(tf.keras.layers.Dense(len(vocabneg)))
#Displaying the model parameters 
mneg.summary()
#Cross entropy as it is a multiple class classification problem. Using ADAM optimizer for it best performs in case of classification
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
mneg.compile(optimizer='adam', loss=loss)


#Desiging our model - Positive
mpos = tf.keras.Sequential()
#An embedding layer to capture meaning of words in a real valued space.
mpos.add(tf.keras.layers.Embedding(len(vocabpos), embedding_dim,
                          batch_input_shape=[BATCH_SIZE, None]))
#Recurrent layer - Long Short Term Memory
mpos.add(tf.keras.layers.LSTM(rnn_units,
                    return_sequences=True,
                    stateful=True,
                    recurrent_initializer='glorot_uniform'))
#Final output layer with one neuron per vovab entry
mpos.add(tf.keras.layers.Dense(len(vocabpos)))
#Displaying the model parameters 
mpos.summary()
#Cross entropy as it is a multiple class classification problem. Using ADAM optimizer for it best performs in case of classification
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
mpos.compile(optimizer='adam', loss=loss)


#Desiging our model - ALL
mall = tf.keras.Sequential()
#An embedding layer to capture meaning of words in a real valued space.
mall.add(tf.keras.layers.Embedding(len(vocaball), embedding_dim,
                          batch_input_shape=[BATCH_SIZE, None]))
#Recurrent layer - Long Short Term Memory
mall.add(tf.keras.layers.LSTM(rnn_units,
                    return_sequences=True,
                    stateful=True,
                    recurrent_initializer='glorot_uniform'))
#Final output layer with one neuron per vovab entry
mall.add(tf.keras.layers.Dense(len(vocaball)))
#Displaying the model parameters 
mall.summary()
#Cross entropy as it is a multiple class classification problem. Using ADAM optimizer for it best performs in case of classification
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
mall.compile(optimizer='adam', loss=loss)



# Directory where the checkpoints will be saved for Negative, Positive and ALL
checkpoint_dirneg = my_temp_folder+'training_checkpoints_lstmneg'
import shutil
try:
    shutil.rmtree(checkpoint_dirneg)
except:
    print("directory not used yet.")
checkpoint_dirpos = my_temp_folder+'training_checkpoints_lstmpos'
import shutil
try:
    shutil.rmtree(checkpoint_dirpos)
except:
    print("directory not used yet.")
checkpoint_dirall = my_temp_folder+'training_checkpoints_lstmall'
import shutil
try:
    shutil.rmtree(checkpoint_dirall)
except:
    print("directory not used yet.")


# Setting up temporary checkpoints for Negative, Positive and ALL
# Creating callback object that can be supplied to the keras fit function. 
# Name of the checkpoint files
checkpoint_prefixneg = os.path.join(checkpoint_dirneg, "ckpt_")
checkpoint_prefixpos = os.path.join(checkpoint_dirpos, "ckpt_")
checkpoint_prefixall = os.path.join(checkpoint_dirall, "ckpt_")

cbneg=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefixneg,
    monitor='loss',
    save_weights_only=True,
    save_best_only=True)

cbpos=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefixpos,
    monitor='loss',
    save_weights_only=True,
    save_best_only=True)

cball=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefixall,
    monitor='loss',
    save_weights_only=True,
    save_best_only=True)


# In[ ]:


#Fitting model for Negative, Positive and ALL
historyneg = mneg.fit(datasetneg, epochs=20,callbacks=[cbneg],verbose=1)
print("training complete for Negative Reviews.")


# In[ ]:


#Loss values for Negative, Positive and ALL
historyneg.history.values() 
historyneg.history.keys() 

#Plotting Negative, Positive and ALL Reviews plots
import pandas as pd
lossdfneg=pd.DataFrame(historyneg.history)
lossdfneg.plot()


# In[ ]:


#These models can now be used to pass through a complete batch of input data and get the outputs
#However, we are reloading the save models and building them with assume a batch input size of 1 only
#Performing these operations for Negative, Positiv and ALL by loading respective checkpoint files
modelneg = tf.keras.Sequential()
modelneg.add(tf.keras.layers.Embedding(len(vocabneg), embedding_dim,
                          batch_input_shape=[1, None]))
modelneg.add(tf.keras.layers.LSTM(rnn_units,
                    return_sequences=True,
                    stateful=True,
                    recurrent_initializer='glorot_uniform'))
modelneg.add(tf.keras.layers.Dense(len(vocabneg)))
#Reloading and changing batch size to 1
modelneg.load_weights(tf.train.latest_checkpoint(checkpoint_dirneg))
modelneg.build(tf.TensorShape([1, None]))

#Generating Text for Negative,Positive and ALL with same seed word as "The "
#Lets generate five sentences, so that i can calculate copus BLEU score
neg_gen_5 = []
for i in range(5):
  neg_text = generate_text(modelneg, start_string=u"The ",chr2=char2idxneg, idx2=idx2charneg)
  print(i," : ",neg_text)
  neg_gen_5.append(nltk.tokenize.word_tokenize(neg_text))


# In[ ]:


#Fitting model for Positive reviews
historypos = mpos.fit(datasetpos, epochs=20,callbacks=[cbpos],verbose=1)
print("training complete for Positive Reviews.")


# In[ ]:


#plottingg loss for Positive reviews
historypos.history.values() 
historypos.history.keys() 

import pandas as pd
lossdfpos=pd.DataFrame(historypos.history)
lossdfpos.plot()


# In[ ]:




modelpos = tf.keras.Sequential()
modelpos.add(tf.keras.layers.Embedding(len(vocabpos), embedding_dim,
                          batch_input_shape=[1, None]))
modelpos.add(tf.keras.layers.LSTM(rnn_units,
                    return_sequences=True,
                    stateful=True,
                    recurrent_initializer='glorot_uniform'))
modelpos.add(tf.keras.layers.Dense(len(vocabpos)))
#Reloading and changing batch size to 1
modelpos.load_weights(tf.train.latest_checkpoint(checkpoint_dirpos))
modelpos.build(tf.TensorShape([1, None]))

#Generating Text for Negative,Positive and ALL with same seed word as "The "
#Lets generate five sentences, so that i can calculate copus BLEU score
pos_gen_5 = []
for i in range(5):
  pos_text = generate_text(modelpos, start_string=u"The ",chr2=char2idxpos, idx2=idx2charpos)
  print(i," : ",pos_text)
  pos_gen_5.append(nltk.tokenize.word_tokenize(pos_text))


# In[ ]:


historyall = mall.fit(datasetall, epochs=20,callbacks=[cball],verbose=1)
print("training complete for ALL Reviews.")


# In[ ]:



historyall.history.values() 
historyall.history.keys() 


import pandas as pd
lossdfall=pd.DataFrame(historyall.history)
lossdfall.plot()


# In[ ]:



modelall = tf.keras.Sequential()
modelall.add(tf.keras.layers.Embedding(len(vocaball), embedding_dim,
                          batch_input_shape=[1, None]))
modelall.add(tf.keras.layers.LSTM(rnn_units,
                    return_sequences=True,
                    stateful=True,
                    recurrent_initializer='glorot_uniform'))
modelall.add(tf.keras.layers.Dense(len(vocaball)))
#Reloading and changing batch size to 1
modelall.load_weights(tf.train.latest_checkpoint(checkpoint_dirall))
modelall.build(tf.TensorShape([1, None]))

#Generating Text for Negative,Positive and ALL with same seed word as "The "
#Lets generate five sentences, so that i can calculate copus BLEU score
all_gen_5 = []
for i in range(5):
  all_text = generate_text(modelall, start_string=u"The ",chr2=char2idxall, idx2=idx2charall)
  print(i," : ",all_text)
  all_gen_5.append(nltk.tokenize.word_tokenize(all_text))


# ## **Evaluation using BLEU Scores**

# In[ ]:


all_neg_reviews


# In[ ]:


#print(all_neg_reviews[0:10]) 
#print(nltk.tokenize.word_tokenize(all_neg_reviews)[0:10])
all_neg_reviews_tokenize = nltk.tokenize.word_tokenize(all_neg_reviews)
all_pos_reviews_tokenize = nltk.tokenize.word_tokenize(all_pos_reviews)
all_reviews_tokenize = nltk.tokenize.word_tokenize(all_reviews)


# In[ ]:


#For more robust numbers i can calculate 5 values for BELU1,BELU2, BELU3, BELU4 for every senetence out of 5 sentences generated and then average the value for reporting
#But this approach is very computation intensive as negative corpus size is large 
# cumulative BLEU scores - For Negative Reviews
from nltk.translate.bleu_score import sentence_bleu
from statistics import mean 
BLEU1_neg = []
BLEU2_neg = []
BLEU3_neg = []
BLEU4_neg = []
for i in range(5):
    reference = all_neg_reviews_tokenize 
    candidate = neg_gen_5[i] 
    BLEU1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    print('Cumulative 1-gram: %f' % BLEU1)
    BLEU1_neg.append(BLEU1)
    BLEU2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
    print('Cumulative 2-gram: %f' % BLEU2)
    BLEU2_neg.append(BLEU2)
    BLEU3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
    print('Cumulative 3-gram: %f' % BLEU3)
    BLEU3_neg.append(BLEU3)
    BLEU4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
    print('Cumulative 4-gram: %f' % BLEU4)
    BLEU4_neg.append(BLEU4)

print(BLEU1_neg,BLEU2_neg,BLEU3_neg,BLEU4_neg)
print(mean(BLEU1_neg),mean(BLEU2_neg),mean(BLEU3_neg),mean(BLEU4_neg))


# In[ ]:


#For one Negative generated review - values of for BELU1,BELU2, BELU3, BELU4 respectively
from nltk.translate.bleu_score import sentence_bleu
reference = all_neg_reviews_tokenize
candidate = neg_gen_5[0] 
print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))


# In[ ]:


#For one Positive generated review - values of for BELU1,BELU2, BELU3, BELU4 respectively 
from nltk.translate.bleu_score import sentence_bleu
reference = all_pos_reviews_tokenize
candidate = pos_gen_5[0] 
print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))


# In[ ]:


#For one "all" generated review - values of for BELU1,BELU2, BELU3, BELU4 respectively 
from nltk.translate.bleu_score import sentence_bleu
reference = all_reviews_tokenize
candidate = all_gen_5[0] 
print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))


# ### Statistical Model for Language generation
# Generating for all reviews

# In[ ]:


get_ipython().system(' pip install nltk')
get_ipython().system(' pip install --upgrade nltk')
import nltk
from nltk.util import ngrams
from nltk.lm import MLE
from nltk import word_tokenize
# we need to download a special component that is used by the tokenizer below -- don't worry about it. 
nltk.download('punkt')


# In[ ]:


a = data_kaggle.review.apply(word_tokenize)

print(a[0:5])
a_revs = []
for review in a:
  a_revs.append(review)
print(a_revs[0:5])

#TAking subset of data as this is highly memory intensive analysis
extract = 0.2
a_revs = a_revs[:int(extract*len(a_revs))]


# In[ ]:


#Preparing train and vocab set for the model to fit on
from nltk.lm.preprocessing import padded_everygram_pipeline
train, vocab = padded_everygram_pipeline(3, a_revs)

#Reviewing sample n-grams 
print(list(ngrams(a_revs[0], n=1)))
print(list(ngrams(a_revs[0], n=2)))
print(list(ngrams(a_revs[0], n=3)))

#Stating the model
model = MLE(3) 
#Fitting over all reviews
model.fit(train, vocab)

#Finally, text generation by our model
word_list = model.generate(15, random_seed=3)
print("Sentences : ",' '.join(word for word in word_list))
print("Tokenized form : ",word_list)


# In[ ]:


#For one Statistically generated review - values of for BELU1,BELU2, BELU3, BELU4 respectively 
from nltk.translate.bleu_score import sentence_bleu
reference = a_revs
candidate = word_list
print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))


# ### Thus, highest BLEU scores are shown by my Statistical model. This can be attributed to limited size of my neural models due to computational complexity plus i did not let my model learn beyond 20 epochs.
# Thus, saving my MLE model as we are asked in the assignment to save the best performing model.

# In[ ]:


get_ipython().system(' pip install h5py')
#from tensorflow.keras.models import save
from google.colab import drive
drive.mount('/content/gdrive')
#Using model.save() to save the weights of the model along with architecture in HDF5 format
#model_lstm.save("/content/gdrive/My Drive/Colab Notebooks/model_lstm_d18129636.h5")
#print("Saved model to mounted google-drive")


# In[ ]:


import pickle

save_model_MLE = open("/content/gdrive/My Drive/Colab Notebooks/TEXT_GENERATION_MLE_D18129636.pickle","wb")
pickle.dump(model, save_model_MLE)
save_model_MLE.close()


# In[ ]:


model_mle = open("/content/gdrive/My Drive/Colab Notebooks/TEXT_GENERATION_MLE_D18129636.pickle", "rb")
classifier = pickle.load(model_mle)
model_mle.close()

