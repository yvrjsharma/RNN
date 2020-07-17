# PART 1: Text classification tasks 

![](/images/alina-grubnyak-ZiQkhI7417A-unsplash.jpg)

Here i am doing sentiment analysis on IMDB movie review dataset with 50,000 entries using Tensorflow2.0 with Keras. 
I have created several models and covered wide variety of topics here - 
1. Sentiment Analysis using single LSTM layer
2. Sentiment Analysis using multiple LSTM layers
3. Sentiment Analysis using simple RNN layers
4. Sentiment Analysis using Pre Trained word embeddings
5. Sentiment Analysis using word embeddings on-the-fly
6. Sentiment Analysis using CNN for Text classifications
7. Sentiment Analysis using CNN plus LSTM layers for Text classifications
8. How to save a deep learning model 
9. Transfer learning on smaller text dataset (30 movie reviews)
10. Text Generation using both LSTM and statistical methods
11. Comparing statistical and deep learning text generation model performances using BLEU scores.

# IMDB Modelling Task
## 1.	Comparison LSTM and RNN
LSTM is widely preferred over RNN because the former’s architecture overcomes the problem of vanishing and exploding gradient problems, common to all RNNs. This allows the LSTMS to grow very large or deep thus able to learn the features hidden in a text more appropriately.

### DROPOUT
I have used Dropout as a computationally cheap and very effective regularization method. It helped in reducing overfitting in my deep neural network and thus improved my model’s generalization ability. In Dropout, during training some member outputs are ignored randomly, known as dropping out of these connections. I am using 0.5 dropout rate in hidden layers and 0.2 for input layers in all my networks in the report. The reasoning behind using these values is as follows –
A common practice is of using 0.5 dropout value or retention value for hidden units and a value between 0.5 to 1 for input layer. Dropout is found to be very effective in text-classification problems. Due to dropout the activations of the hidden unit become sparse or thin and as such a wider network or more nodes may be required while using dropout. If we want to use dropout regularization in a network, then the number of units in the original network should be increased by one over the dropout probability used or 1/p. So if p is 0.5, then the new number of units after applying dropout should be increased by factor of 2 or should be doubled (Srivastava, Hinton, Krizhevsky, Sutskever, and Salakhutdinov, 2014).
Also, 50% dropout rates for hidden layers while 20% for visible input layers is advised by another widely cited paper (Hinton, Srivastava, Krizhevsky, Sutskever, and Salakhutdinov, 2012).
Also note that in Tensorflow, dropout rate of 1 would mean 0 outputs, while a rate of 0 would mean 100 percent outputs. So as rate approaches 1 less and less units are output.

### Early Stopping
It is another way of stopping over fitting in your deep neural networks specially in RNN networks. Early stopping produces a good baseline in RNN based models (Gal and Ghahramani, 2016).
It is a callback process and while using it you need to specify a performance measure that is needed to be monitored, the mode or direction in which it is monitored for example minimum or maximum, and lastly a stopping criteria called patience value. I am using early stopping as a regularization criteria in all my models and keeping patience value as 20 because it is optimum for a validation loss cycle to attain its global minima.
The validation error usually while going down would go up slightly in between and then resume downward motion. A Validation error curves can have more than one minimum locally (Prechelt, 2012)

### Epochs
I am using 100 epochs to train all my networks in the report. Using early stopping to minimize the validation loss has allowed me to keep my epoch value as high as 100.

## Discussion: Experiments
For deciding upon the usage of dropout layers in LSTM implementation while using Word Embeddings that were learned on-the-fly, I conducted two small experiments. I am using 50% data or 25000 instances for training, 30% data or 15000 instances for validation while 20% data or 10000 instances for testing. Also note that I am conducting a cross validation examination for all my models by shuffling and splitting data into training-validation-test over five times and then averaging the validation accuracy before reporting. One more step that I have taken in building a robust model is that I have used stratifying approach in splitting data in train-validation-test segments, meaning the original distribution of negative and positive is maintained in every subset of data that is used. More on this can be seen in code files submitted along with the report.
1.	In first experiment, my hypotheses was that training an LSTM network on 50% imdb dataset instances, using single dropout layer on hidden layers, with dropout rate as 0.5, I would get higher accuracy over the validation dataset of 30% imdb dataset instances, then when no dropout layer is used. 
2.	In second experiment- my hypotheses was that training an LSTM network on 50% imdb dataset instances, using one dropout layer on hidden layers, with dropout rate as 0.5, and another dropout layer with rate as 0.2 on input layer, I would get higher accuracy over the validation dataset of 30% imdb dataset instances, then when no dropout layer is used. 

### Results
With no dropout layer was used, the mean validation accuracy on 30% validation data was 85.8, test accuracy on 20% hold-out imdb data was 86.01, while training clearly showed signs of overfitting with accuracy at 99.82% on 50% training data. When dropout layer with rate 0.5 was used on hidden RELU layer,  the mean validation accuracy on 30% validation data was 85.77, test accuracy on 20% hold-out imdb data was 85.76, while training still showed signs of overfitting with accuracy at 98.6% on 50% training data. Lastly, when dropout layer with rate 0.5 was used on hidden RELU layer, and another dropout layer with rate as 0.2 was used on input layer,  the mean validation accuracy on 30% validation data was 86.23, test accuracy on 20% hold-out imdb data was 87.31, while training still showed signs of overfitting with accuracy at 99.28 on 50% training data (Figure 2 in Appendix-1). All accuracies terms are in percentages.
### Discussion
In the interest of time and GPU resources, I have not conducted t-tests on the distribution of validation accuracies I have gathered by doing cross validation over five train-validation datasets. However, to deduce which of the training accuracies are higher, by doing cross validation and reporting the mean validation accuracy thus achieved, I can conclude from my experiments robustly enough that the model with dropout layers on both hidden and input layers performed the best.

Similar experiments with and without dropout layers were conducted for following models and dropout increased the validation accuracy marginally in every case (except in case of RNN one). The best Text Classification model, according to my Mean Validation Accuracy is the model with Single Layer LSTM and Embeddings learned on the fly –
1.	Simple RNN over on-the-fly learned word embeddings (Figure 3 without dropout, Figure 4 with dropout as 0.5 on hidden layer and 0.2 on input layer), 
2.	Multi-layer LSTM with word embeddings learned on-the-fly (Figure 5 without dropout, Figure 6 with dropout as 0.5 on hidden layer and 0.2 on input layer),
3.	Sentiment analysis using Pre-trained word embeddings and Dense RELU (Figure 7 without dropout, Figure 8 with dropout as 0.5 on hidden layer and 0.2 on input layer),
4.	Sentiment analysis using word embeddings learned on-the-fly and Dense RELU (Figure 9 without dropout, Figure 10 with dropout as 0.5 on hidden layer and 0.2 on input layer),
5.	CNN for Text classification (Figure 11 without dropout) 

## 2.	Multi-layer LSTM 
It was done using the same architecture as single layer LSTM and the validation accuracy was not improved even marginally.

## 3.	Embeddings
I have created two models – one with Pre-trained word embeddings and another one with word embeddings learned on the fly.
1.	Sentiment analysis using Pre-trained word embeddings and Dense RELU
Getting the dataset in form of csv from Kaggle using Kaggle API. In this step the reason I am using using data downloaded from kaggle and not the one available from Keras is that I need to pass string tensors as an input to the pre-trained word embedding that I am using from the Tensorflow Hub. The IMDB dataset available on Keras is a sequence of numbers and not words and hence not using it in this specific model. Please note that the two datasets are completely identical.
2.	Sentiment analysis using word embeddings learned on the fly and Dense RELU
Dataset is from tensorflow_datasets

## Discussion: Model Comparisons 
For all my models in the Jupyter notebook submitted, I have calculated 5-fold cross validation accuracies. Mentioning the accuracies here in form of a table for an easier comparison from my latest run –
Model Name |	Accuracy (Mean +/- Std. Dev.)
-----------|-------------------------------
LSTM	| 86.05
Multi-layer LSTM	| 85.68
Simple RNN	| 78.29
Pre-trained Embeddings	| 78.43
On-the-fly embeddings	| 86.37
CNN(Multiple and Heterogeneous kernels)	| 89.12
CNN+LSTM	| 85.29

## Saving the Model
I have saved my Deep learning Models' Weights Architecture into a single H5 file. This way of saving a model saves following attributes about a model to a file (model.h5):
1.	Weights learned by a model
2.	Model Layer Architecture
3.	Model loss and metrics
4.	Model optimizer properties
Tensorflow2.0 Keras allows us to load and use this model directly from the saved file. I have performed this operation too in my codes submitted as a part of assignment.


# PART-2: Transfer Learning

1.	Transfer learning has several benefits like – it speeds up the learning, les data is needed and lastly we can use the best model available by fine-tuning them.
2.	There are TWO types of Transfer Learning methodologies:
- Weight Initialization - if all the network weights can change or adapt to the new dataset, then transfer learning simply serves as a weight initialization method.
- Feature Extraction Method - if only weights in output layer is allowed to change while all hidden layer's weights are kept fixed, then transfer learning serves as a feature extraction method.
3.	In my analysis, I have performed transfer learning while keeping 0, 1,2,3,4 and all 5 rows as frozen for source model. My analysis proved that with first 3 layers frozen transfer learning produced highest validation accuracy, I have provided a comparison of all validation accuracies and approaches in my submitted notebook file.
4.	As expected in transfer learning, the training took very less time in terms of the learning curve in all the cases as shown in notebook. Our new model used the weights from older saved best model, fitted on a larger dataset from a related problem. Along with increased learning speed, this transfer learning also resulted in lower generalization error as is evident from the validation scores.
5.	The best model has accuracy and minimum standard deviation as - 88.89% +/- 9.94%
6.	I have already Saved the model thus learned and provided as part of assignment submission.

### Working with my own Data
1.	I have constructed a labelled dataset in form of a .csv file. The dataset has three variables - Review, Sentiment, and Movie. This dataset is freshly created from the actual IMDB movie database from https://www.imdb.com/search/title/ by taking 30 movies from my birth year 1987. For each movie, I have selected one 'Positive' and one 'Negative' review and all the reviews are in English. Also note that, I have included the titles of the reviews as well in the review text. I am supplying this raw data of 60 movie reviews along with my source code.
2.	I have prepared "From Scratch" model on this 30 Movie database using exactly same architecture as my best saved model which is the LSTM model with word embeddings learned on the fly. The accuracy came very low (around 60%) 
3.	Initially I expected the reason of low accuracy was my large vocab size or my larger max word length from my “best” model, but my further analysis proved that even lower vocab and word length gave similar accuracy readings. Codes for this analysis is in the notebook I have submitted as part of my assignment.
4.	This model served as a baseline model for performance. I investigated how the addition of transfer learning affected the performance on this problem and the mean validation set accuracy was increased to 90%. Please see my notebook for a very detailed analysis.

# PART-3: Text Generation using Neural implementation

1.	The specific problem of Language Modeling is learning what makes a language a language so that we can predict the next word given a sequence of words. Goodness of a text generation model is measured in terms of Perplexity. Perplexity is a length normalized inverse probability of a sentence given a specific language model. When building language models, the ones with smaller Perplexity measures are termed as better ones. Thus we need to reduce the average Perplexity over our data as we train our model. 
2.	As a text corpus I am importing IMDB large movie reviews dataset using Kaggle API. Dataset consist of 50000 reviews equally divided into negative and positive reviews. First, I have split data into two parts, one part has all the negative reviews while the second part has all the positive ones. Also, a separate copy of all the reviews is created too. As an input requirement I concatenated all the reviews into strings of positive, negative and combined reviews first. A vocab is created out all these three strings. 
3.	I am implementing the text generation using LSTM network and in LSTM architecture, there are two outputs from every RNN unit – one is the central hidden state while other one is output fed from one time state to another. In our language generation case, the predicted word will be fed in as input to next state to generate the next word. I am keeping seed length as one (by keeping the Batch_size as one) as it makes it easier for us to run one example at a time.
4.	I have created separate vocabs of all characters available inside individual corpus of negative, positive an all reviews. I am creating an embedding layer in my model and passing the vocabulary size to it along with the sequence length and embedding dimension as 256. I have one LSTM hidden layer with 2048 as its state size. It is followed up by an output layer of linear activation function, our cost function is set up to deal with linear activations, or logits. Sparse categorical crossentropy loss function is suitable for multi-class classification along with ADAM as gradient descent method. Model are fitted with batch size as 64 for 20 training epochs for best model learning and also in the interest of time. Checkpoints are created to monitor loss while fitting the model on datasets and save the best model or the model with lowest loss function values automatically. As seen during code implementation, the model learning is very slow even on strongest high-ram Colab Pro GPUs. Checkpoints allows us to later run our model at any point without needing to run the whole batch. 
5.	After training my models on smaller extracted data, rather than using the model to pass through a complete batch of input data and get the outputs,  I am reloading the saved models and building by calling the build function with a batch input size of 1 only. This makes it easier to run one example at a time through the already trained network.

### Writing my own Reviews - Text Generation using statistical model
1.	A statistical language model is built using ngram methods, in which a body of text is parsed and tables of all 1-grams, 2-grams, 3-grams and perhaps more are build up. Further probability calculations give probabilities for individual words given a context. Frequency of words and of sequences of words is an important statistics in any nGram based statistical approach. 2-gram means a sequence of 2 words, 3-gram means sequence of three words and so on. NLTK has funcions which can calculate such frequencies 
2.	Python NLTK (natural langauge toolkit) has a simple implementation of an nGram Language Modeler, called MLE and I am using it to build my language models from negative, positive and all review texts. These reviews are from IMDB movie database of 50,000 reviews which are labelled as positive and negative. 
3.	Just like in convolutional neural networks we include a padding to make sure starting and ending pixels get a representation in the calculations, there is a concept of padding in ngram approach too. NLTK can take care of the padding. 
4.	After padding we have with us train and vocab generator objects, which are nothing but lists of n-grams and the vocab list for the complete text respectively. Now we fit the model on the train and vocab objects and generate text based on random seed values.

## Discussion: Evaluation
1.	The Bilingual Evaluation Understudy or BLEU score, is a metric for automatic evaluation of machine translation or any sentence generation task. It basically, compares n-grams of the target and source and count the number of matches. More matches would mean better translation or better performing generator (Papineni, Roukos, Ward, and Zhu, 2002)
2.	It is an accepted norm to report the BLEU-1 to BLEU-4 scores for describing the performance of text generation system. Here 1 and 4 represents 1-gram and 4-gram words respectively and scores are Cumulative N-Gram Scores. These scores refer to the calculation of BLEU-1 to BLEU-4 n-gram scores by using the weighted geometric mean as weighting method.
3.	Python NLTK library provides sentence_bleu() and corpus_bleu() functions to calculate scores at individual sentence level or at corpus level (a group or a body of sentences) respectively.
4.	My calculations for the BLEU-1 to BLEU-4 has suggested that the statistical model produced the best results in text generation task. 
5.	Highest BLEU scores are shown by the Statistical model because of the limited size of my neural language models and due to computational complexity and limitations it was not feasible to run the neural models for much higher epochs.

## References
1.	Srivastava, Nitish, Hinton, Geoffrey, Krizhevsky, Alex, Sutskever, Ilya, and Salakhutdinov, Ruslan. Dropout: A simple way to prevent neural networks from overfitting. J. Mach. Learn. Res., 15(1):1929–1958, January 2014. Retrieved from http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
