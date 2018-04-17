# Author: Laura Wendlandt
# Dependencies: sklearn
# Do simple analysis of the ridge regression model

import pickle
from sklearn.linear_model import Ridge

# @TODO Change these variables before running
dataFolder = '/local/embedding_datasets/metaClassifier/' #Base path
MODEL_FILE = dataFolder+'regressionModel.pkl' #Saved ridge regression model
TRAINING_DATA_FILE = dataFolder+'X.pkl' #Training data from regression model
TRAINING_TARGET_FILE = dataFolder+'y.pkl' #Labels for the training data from regression model

print("Loading regression model...")
with open(MODEL_FILE,'rb') as pickleFile:
    model = pickle.load(pickleFile)

print("Loading training data...")
with open(TRAINING_DATA_FILE,'rb') as pickleFile:
    X = pickle.load(pickleFile)
with open(TRAINING_TARGET_FILE,'rb') as pickleFile:
	y = pickle.load(pickleFile)

print("Scoring... (*slow*)")
score = model.score(X,y)
print("Model has an R^2 score of " + str(score) + " on the training data.")

weights = model.coef_
#labels
labels = []
#posSenseFeatures
labels.append('Primary POS = adjective')
labels.append('Primary POS = adposition')
labels.append('Primary POS = adverb')
labels.append('Primary POS = conjunction')
labels.append('Primary POS = determiner')
labels.append('Primary POS = noun')
labels.append('Primary POS = number')
labels.append('Primary POS = particle')
labels.append('Primary POS = pronoun')
labels.append('Primary POS = verb')
labels.append('Primary POS = punctuation')
labels.append('Primary POS = other')
labels.append('Secondary POS = adjective')
labels.append('Secondary POS = adposition')
labels.append('Secondary POS = adverb')
labels.append('Secondary POS = conjunction')
labels.append('Secondary POS = determiner')
labels.append('Secondary POS = noun')
labels.append('Secondary POS = number')
labels.append('Secondary POS = particle')
labels.append('Secondary POS = pronoun')
labels.append('Secondary POS = verb')
labels.append('Secondary POS = punctuation')
labels.append('Secondary POS = other')
labels.append('Number of different WordNet POS')
labels.append('Number of different WordNet senses')
#syllableFeatures
labels.append('Number of syllables (0 if unknown)')
#frequencyFeatures
labels.append('Higher raw freq. of word in corpora')
labels.append('Lower raw freq. of word in corpora')
labels.append('Abs. difference in raw freq. of words corpora')
#vocabSizeFeatures
labels.append('Higher vocabulary size of corpora')
labels.append('Lower vocabulary size of corpora')
labels.append('Abs. difference in vocabulary size in corpora')
#bagOfDomainsFeatures
labels.append('Number of U.S. Domain')
labels.append('Number of New_York_and_Region Domain')
labels.append('Number of Business Domain')
labels.append('Number of Arts Domain')
labels.append('Number of Sports Domain')
labels.append('Number of NYT (all) Domain')
labels.append('Number of Europarl Domain')
labels.append('Do the two domains match?')
#vocabOverlapFeatures
labels.append('Overlap between corpora vocabulary')
#trainingDataPositionsFeatures
labels.append('Lower training data position')
labels.append('Higher training data position')
labels.append('Abs. difference in training data position')
#bagOfAlgorithmsFeatures
labels.append('Number of w2v')
labels.append('Number of ppmi')
labels.append('Number of glove')
labels.append('Do the two algorithms match?')
#algorithmDimensionFeatures
labels.append('Higher embedding dimension for algorithms')
labels.append('Lower embedding dimension for algorithms')
labels.append('Abs. difference in embedding dimension for algorithms')
labels.append('Do the two dimensions match?')

print("Here are the weights of the model:")
for (weight,feature) in sorted(zip(weights,labels)):
	print(weight,feature)
