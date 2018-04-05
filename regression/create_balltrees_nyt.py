# Author: Laura Wendlandt
# Dependencies: sklearn, numpy
# Create balltrees for all embedding spaces
# Note: to reproduce the regression model, this script must be run 6 different times (one for each different NYT domain)

import numpy as np
import pickle
import time
import math
from sklearn.neighbors import BallTree
from collections import Counter

# @TODO Change these variables before running
WORDLIST_FILE = '/local/embedding_datasets/nyt_corpus/wordList.pkl' #Location of pickled wordlist (see getWordList.py)
NYT_DOMAIN = 'all' #Which NYT domain to train embedding spaces for
NYT_FILE = '/local/embedding_datasets/nyt_corpus/processed/top5.data' #NYT domain data, one sentence per line, already tokenized with tokens separated by spaces
MODEL_FOLDER = '/local/embedding_datasets/nyt_corpus/metaClassifier/' #Folder where all of the trained models are
OUTPUT_FOLDER = MODEL_FOLDER #Folder to save the balltrees and distance matrices
LEAF_SIZE = 2 #Parameter for balltrees, affects speed of balltrees (but not accuracy)

print('Prepare all training data...')
with open(NYT_FILE,'r') as domainFile:
    sentences = domainFile.readlines()
    sentences = [[word for word in i[:-1].split(' ')] for i in sentences] #tokenize on space

print('Load all models...')
names = []
all_models = []
for algorithm in ['w2v','glove','ppmi']:
    for dimension in [50,100,200,400,800]:
        if algorithm == 'ppmi':
            name = NYT_DOMAIN + '_' + algorithm + '_' + str(dimension)
            with open(MODEL_FOLDER+name+'.pkl','rb') as pickleFile:
                all_models.append(pickle.load(pickleFile))
                names.append(name)
            continue
        for seed in [2518,2548,2590,29,401]:
            name = NYT_DOMAIN + '_' + algorithm + '_' + str(dimension) + '_' + str(seed)
            with open(MODEL_FOLDER+name+'.pkl','rb') as pickleFile:
                all_models.append(pickle.load(pickleFile))
                names.append(name)

print('Load word list...')
with open(WORDLIST_FILE,'rb') as pickleFile:
    wordList = pickle.load(pickleFile)

print('Pre-calculating distances...')
#Pre-calculate distances for all pairs
#This is the super time-consuming part, but it makes the balltrees very fast
for name,model in zip(names,all_models):
    print(name)
    #Pre-calculate inverse magnitudes
    inverseMagnitudes = {}
    for word in wordList:
        inverseMagnitudes[word] = 1.0 / np.linalg.norm(model[word],ord=2)

    #Calculate all distances
    distances = np.zeros((len(wordList),len(wordList))) #1.0 - cos similarity (doubly indexed by position of word in wordList)
    for i in range(len(wordList)):
        for j in range(len(wordList)):
            if i<j:
                continue #Cos similarity is symmetric
            if i==j:
                distances[i][j] = 0.0
                continue
            word1 = wordList[i]
            word2 = wordList[j]
            distances[i][j] = distances[j][i] = 1.0 - (np.dot(model[word1],model[word2]) * inverseMagnitudes[word1] * inverseMagnitudes[word2])

    #Save distance matrix for future use
    if NYT_DOMAIN != 'all':
        with open(OUTPUT_FOLDER + 'distances_'+name+'.pkl','wb') as pickleFile:
            pickle.dump(distances,pickleFile)
    else: #domain == 'all'
        for i in range(1000,len(wordList)+999,1000):
            with open(OUTPUT_FOLDER + 'distances_'+name+'.pkl','wb') as pickleFile:
                pickle.dump(distances[i-1000:i],pickleFile)

    #Define distance function for balltree
    #Note: Distances (matrix doubly indexed by word indices) needs to be defined
    def mydist(x, y):
        return distances[int(x[0])][int(y[0])]

    print('Pre-buid balltrees...')
    balltree = BallTree([[i] for i in range(len(wordList))],metric='pyfunc',func=mydist,leaf_size=LEAF_SIZE) #Leaf size is adjustable

    #Save balltree for future use (always need to define distance func before loading balltrees)
    with open(OUTPUT_FOLDER+name+'_balltree'+'.pkl', 'wb') as pickle_file:
        pickle.dump(balltree,pickle_file)
