# Author: Laura Wendlandt
# Dependencies: numpy
# Train 5 embedding spaces (5 embedding sizes) using PPMI on NYT
# Note: to reproduce the regression model, this script must be run 6 different times (one for each different NYT domain)

import numpy as np
import pickle
from collections import Counter
import math

# @TODO Change these variables before running
WORDLIST_FILE = '/local/embedding_datasets/nyt_corpus/wordList.pkl' #Location of pickled wordlist (see getWordList.py)
NYT_DOMAIN = 'all' #Which NYT domain to train embedding spaces for
NYT_FILE = '/local/embedding_datasets/nyt_corpus/processed/top5.data' #NYT domain data, one sentence per line, already tokenized with tokens separated by spaces
OUTPUT_FOLDER = '/local/embedding_datasets/nyt_corpus/metaClassifier/' #Where the embedding spaces should be saved

def buildCooccurrenceMatrix(sentences,wordList,windowSize=5):
    cooccur = np.zeros((len(wordList)+1,len(wordList)+1)) #Last dimension is for unknown words
    for sentence in sentences:
        sentence = [wordList.index(i) if i in wordList else len(wordList) for i in sentence]
        for i in range(len(sentence)-1):
            for j in range(1,windowSize):
                if i+j >= len(sentence):
                    break
                cooccur[sentence[i]][sentence[i+j]] += 1
                if sentence[i+j] != sentence[i]:
                    cooccur[sentence[i+j]][sentence[i]] += 1
    return cooccur

def getPPMI(cooccur,wordList):
    dMagnitude = 0
    singleCounts = []
    for i in range(len(cooccur)):
        count = 0
        for j in range(len(cooccur)):
            count += cooccur[i][j]
        singleCounts.append(count)
        dMagnitude += count
        
    #Calculate ppmi for all entries in cooccur
    ppmi = np.zeros((len(cooccur),len(cooccur)))
    for i in range(len(cooccur)):
        for j in range(len(cooccur)):
            if i < j: #will be a symmetric matrix 
                break
            if cooccur[i][j] > 0:
                pmi = math.log((cooccur[i][j]*dMagnitude)/(singleCounts[i]*singleCounts[j]),2)
                if pmi > 0:
                    ppmi[i][j] = ppmi[j][i] = pmi
                    
    #Return full SVD
    (U,S,V) = np.linalg.svd(ppmi)
            
    return (U,S)

def train_ppmi(U,S,wordList,embeddingSize,name):
    filenames = []
    
    #Train model
    model = U[:,:embeddingSize].dot(np.diag(S[:embeddingSize]))
    
    #Save model for future use
    filename = name + '_ppmi_' + str(embeddingSize)
    filenames.append(filename)
    save_dict = {}
    for i in range(len(wordList)):
        save_dict[wordList[i]] = model[i]
    with open(OUTPUT_FOLDER+filename+'.pkl', 'wb') as pickle_file:
        pickle.dump(save_dict,pickle_file)
        
    return filenames

print('Loading data...')
with open(NYT_FILE,'r') as europarl:
	sentences = europarl.readlines()
	sentences = [i[:-1].split(' ') for i in sentences]

print('Getting word list...')
with open(WORDLIST_FILE,'rb') as pickleFile:
	wordList = pickle.load(pickleFile)

print('Building cooccurrence matrix...')
cooccur = buildCooccurrenceMatrix(sentences,wordList)
print('Getting PPMI...')
(U,S) = getPPMI(cooccur,wordList)
print('Training PPMI...')
embedding_sizes = [50,100,200,400,800]
for size in embedding_sizes:
	print(size)
	train_ppmi(U,S,wordList,size,NYT_DOMAIN)
