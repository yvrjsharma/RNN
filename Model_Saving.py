
# ## **Model Saving**

# ### Saving my Deep learning Models' Weights Architecture into a single H5 file -
# 
# This way of saving a model saves following attributes about a model to a file (model.h5):
# 1. Weights learned by a model
# 2. Model Layer Architecture
# 3. Model loss and metrics
# 4. Model optimizer properties
# 
# Tensorflow2.0 Keras allows us to load and use this model directly from the saved file. 
# 

# In[ ]:


get_ipython().system(' pip install h5py')
from tensorflow.keras.models import save
from google.colab import drive
drive.mount('/content/gdrive')
#Using model.save() to save the weights of the model along with architecture in HDF5 format
model_lstm.save("/content/gdrive/My Drive/Colab Notebooks/model_lstm_d18129636.h5")
print("Saved model to mounted google-drive")


# In[ ]:


# load model
model_transfer = load_model('/content/gdrive/My Drive/Colab Notebooks/model_lstm_d18129636.h5')


