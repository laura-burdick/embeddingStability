# Author: Laura Wendlandt
# Generate all features

import numpy as np
import pickle
from collections import Counter
import string
import time
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn
from nltk.corpus import cmudict
from sklearn.neighbors import BallTree
from nltk.corpus import brown
import nltk
import sys

# @TODO Change these variables before running
datasetFolder='/local/embedding_datasets/' #Base path
NYT_FOLDER = datasetFolder+'nyt_corpus/processed/' #All NYT domain data is here, one sentence per line, already tokenized with tokens separated by spaces
EUROPARL_FILE = datasetFolder+'europarl/europarl_processed.txt' #All of English Europarl, one sentence per line, already tokenized with tokens separated by spaces
NYT_WORDLIST_FOLDER = datasetFolder+'nyt_corpus/metaClassifier/' #Location of all NYT wordlists (see getWordList.py)
EUROPARL_WORDLIST_FILE = datasetFolder+'europarl/wordList.pkl' #Location of Europarl wordlist
SHARED_WORDLIST_FILE = datasetFolder+'combinedWordList.pkl' #Shared word list (see sharedWordList.py)
NYT_NEAREST_NEIGHBORS_FOLDER = datasetFolder+'nyt_corpus/metaClassifier/tenNearestNeighbors/' #Location of all pre-calculated nearest neighbors for NYT dataset
EUROPARL_NEAREST_NEIGHBORS_FOLDER = datasetFolder+'europarl/metaClassifier/tenNearestNeighbors/' #Location of all pre-calculated nearest neighbors for Europarl dataset
OUTPUT_FOLDER = datasetFolder+'metaClassifier/' #Save all features here

if len(sys.argv) < 3:
    print('This program takes two arguments: a start index (inclusive) and an end index (exclusive) for training')
    exit(0)
start_index = int(sys.argv[1])
end_index = int(sys.argv[2])

def stability(word,similar1,similar2,same=False):
    if same and len(similar1) == 1:
        return len(similar1[0])
    
    sets1 = [set(a) for a in similar1]
    if not same:
        sets2 = [set(b) for b in similar2]
    else:
        sets2 = sets1
    
    avgOverlap = 0
    for i in range(len(similar1)):
        for j in range(len(similar2)):
            if not same or (same and i!=j):
                avgOverlap += len(sets1[i] & sets2[j])

    if same:
        avgOverlap /= (len(similar1)*len(similar2)-len(similar1))
    else:
        avgOverlap /= (len(similar1)*len(similar2))
    return avgOverlap

def frequencyFeatures(counters,domains,domain1,domain2,word): #NOTE: This is raw frequency, not normalized frequency
    features = []
    
    #Higher RAW freq. of word in either corpus A or corpus B
    #Lower RAW freq. of word in either corpus A or corpus B
    #Abs. difference in RAW freq. of words in corpus A and B
    frequency1 = counters[domains.index(domain1)][word]
    frequency2 = counters[domains.index(domain2)][word]
    if frequency1 > frequency2:
        features.append(frequency1)
        features.append(frequency2)
    else:
        features.append(frequency2)
        features.append(frequency1)
    features.append(abs(frequency1-frequency2))
    
    return features

def vocabSizeFeatures(wordlists,domains,domain1,domain2):
    features = []
    
    #Higher vocabulary size of either corpus A or corpus B
    #Lower vocabulary size of either corpus A or corpus B
    #Abs. difference in vocabulary size in corpus A and B
    vocabSize1 = len(wordlists[domains.index(domain1)])
    vocabSize2 = len(wordlists[domains.index(domain2)])
    if vocabSize1 > vocabSize2:
        features.append(vocabSize1)
        features.append(vocabSize2)
    else:
        features.append(vocabSize2)
        features.append(vocabSize1)
    features.append(abs(vocabSize1-vocabSize2))
    
    return features

def bagOfDomainsFeatures(domains,domain1,domain2):
    features = []
    
    #Bag of domains: Is domain 1 represented? Is domain 2 represented? etc. (Each feature is 0, 1, or 2)
    domainsRepresented = [0] * len(domains)
    domainsRepresented[domains.index(domain1)] += 1
    domainsRepresented[domains.index(domain2)] += 1
    for domainRepresented in domainsRepresented:
        features.append(domainRepresented)
        
    #Do the corpora domains match?
    if domain1==domain2:
        features.append(1)
    else:
        features.append(0)
    
    return features

def vocabOverlapFeatures(vocabOverlap,domain1,domain2):
    features = []
    
    features.append(vocabOverlap[(domain1,domain2)])

    return features

def trainingDataPositionsFeatures(trainingDataPositions,word,domains,domain1,domain2):
    features = []

    #Lower trainingDataPosition of either corpus A or corpus B
    #Abs. difference in trainingDataPosition
    pos1 = trainingDataPositions[word][domains.index(domain1)]
    pos2 = trainingDataPositions[word][domains.index(domain2)]
    if pos1 < pos2:
        features.append(pos1)
        features.append(pos2)
    else:
        features.append(pos2)
        features.append(pos1)
    features.append(abs(pos1-pos2))

    return features

def posSenseFeatures(word,possible_pos,wordDists):
    features = []
    
    #Primary POS
    #Secondary POS
    #Number of different POS
    pos_primary = [0] * len(possible_pos)
    pos_secondary = [0] * len(possible_pos)
    num_different = 0
    
    if word in wordDists:
        counter = Counter(wordDists[word])
        primary = counter.most_common()[0][0]
        pos_primary[possible_pos.index(primary)] = 1
        if len(counter.most_common()) > 1:
            secondary = counter.most_common()[1][0]
            pos_secondary[possible_pos.index(secondary)] = 1
        num_different = len(counter.most_common())
                     
    features += pos_primary
    features += pos_secondary
    features.append(num_different)

    #Number of WordNet senses
    synsets = wn.synsets(word)
    features.append(len(synsets))

    return features

def syllableFeatures(cmu_dict,word):
    features = []
    
    #Number of syllables
    #http://www.onebloke.com/2011/06/counting-syllables-accurately-in-python-on-google-app-engine/
    if word.lower() in cmu_dict:
        syllables = [len(list(y for y in x if str.isdigit(y[-1]))) for x in cmu_dict[word.lower()]]
        if len(syllables) > 0:
            features.append(syllables[0])
        else:
            features.append(0)
    else:
        features.append(0)
        
    return features

def algorithmDimensionFeatures(dimension1,dimension2):
    features = []
    
    #Higher embedding dimension for either algorithm A or algorithm B
    #Lower embedding dimension for either algorithm A or algorithm B
    #Abs. difference in embedding dimension for algorithm A and B
    if dimension1 > dimension2:
        features.append(dimension1)
        features.append(dimension2)
    else:
        features.append(dimension2)
        features.append(dimension1)
    features.append(abs(dimension1-dimension2))
    
    #Do the embedding dimensions match?
    if dimension1==dimension2:
        features.append(1)
    else:
        features.append(0)
        
    return features

def bagOfAlgorithmsFeatures(algorithm1,algorithm2):
    features = []
    
    #Bag of algorithms: Is algorithm 1 represented? Is algorithm 2 represented? etc. (Each feature is 0, 1,
    #or 2)
    algorithms = ['w2v','ppmi','glove']
    algorithmsRepresented = [0] * len(algorithms)
    algorithmsRepresented[algorithms.index(algorithm1)] += 1
    algorithmsRepresented[algorithms.index(algorithm2)] += 1
    for algorithmRepresented in algorithmsRepresented:
        features.append(algorithmRepresented)
        
    #Do the algorithms match?
    if algorithm1==algorithm2:
        features.append(1)
    else:
        features.append(0)
        
    return features

def corpusFeatures(trainingDataPositions,vocabOverlap,counters,domains,domain1,domain2,word,wordlists):
    features = []
   
    features = features + frequencyFeatures(counters,domains,domain1,domain2,word) #NOTE: this is raw frequency, not normalized frequency
    features = features + vocabSizeFeatures(wordlists,domains,domain1,domain2)
    features = features + bagOfDomainsFeatures(domains,domain1,domain2)
    features = features + vocabOverlapFeatures(vocabOverlap,domain1,domain2)
    features = features + trainingDataPositionsFeatures(trainingDataPositions,word,domains,domain1,domain2)

    return features

def wordFeatures(word,possible_pos,cmu_dict,wordDists):
    features = []
    
    features = features + posSenseFeatures(word,possible_pos,wordDists)
    features = features + syllableFeatures(cmu_dict,word)
    
    return features

def algorithmFeatures(algorithm1,algorithm2):
    features = []
    
    features = features + bagOfAlgorithmsFeatures(algorithm1,algorithm2)
    
    return features

print('Load CMU dictionary...')
#Load CMU dictionary (needed for feature extraction)
cmu_dict = cmudict.dict()

domains = ['U.S.','New_York_and_Region','Business','Arts','Sports']

print('Prepare all training data...')
#Prepare all training data
data = {}
wordlists = []
for domain in domains:
    print(domain)
    with open(NYT_FOLDER+domain+'.data','r') as domainFile:
        sentences = domainFile.readlines()
        data[domain] = [i[:-1].split(' ') for i in sentences] #tokenize on space
    with open(NYT_WORDLIST_FOLDER+domain+'_wordList.pkl','rb') as pickleFile:
        wordlists.append(pickle.load(pickleFile))

print('all')
domains.append('all')
with open(NYT_FOLDER+'top5.data','r') as domainFile:
    sentences = domainFile.readlines()
	data['all'] = [i[:-1].split(' ') for i in sentences]
with open(NYT_WORDLIST_FOLDER+'all_wordList.pkl','rb') as pickleFile:
    wordlists.append(pickle.load(pickleFile))

print('Europarl')
domains.append('Europarl')
with open(EUROPARL_FILE,'r') as europarl:
    sentences = europarl.readlines()
    data['Europarl'] = [i[:-1].split(' ') for i in sentences]
with open(EUROPARL_WORDLIST_FILE,'rb') as pickleFile:
    wordlists.append(pickle.load(pickleFile))

print('Loading shared word list...')
with open(SHARED_WORDLIST_FILE,'rb') as pickleFile:
    words = pickle.load(pickleFile)

print('Count words in each domain...')
#Count the number of words in each domain (needed for feature extraction)
counters = []
for domain in domains:
    counter = Counter()
    for sentence in data[domain]:
        for word in sentence:
            counter[word] += 1
    counters.append(counter)

#Pre-compute vocab overlap features
print('Pre-compute vocab overlap features...')
vocabOverlap = {}
vocabs = []
for i in range(len(domains)):
    vocabs.append(list(counters[i]))
for i in range(len(domains)):
    for j in range(len(domains)):
        if i<j: #only do top half of matrix
            continue
        if i==j:
            vocabOverlap[(domains[i],domains[j])] = 1.0
            continue
        vocab1 = vocabs[i]
        vocab2 = vocabs[j]
        totalVocabSize = len(set(vocab1+vocab2))
        overlap = len(set([i for i in vocab1 if i in vocab2]))
        vocabOverlap[(domains[i],domains[j])] = vocabOverlap[(domains[j],domains[i])] = float(overlap) / float(totalVocabSize)

print('Get all POS distributions...')
#Get all possible POSs (needed for feature extraction)
possible_pos = ['ADJ','ADP','ADV','CONJ','DET','NOUN','NUM','PRT','PRON','VERB','.','X'] #Universal tagset
brown_tagged = brown.tagged_words(tagset='universal')
wordDists = nltk.ConditionalFreqDist((word.lower(), tag) for (word, tag) in brown_tagged)

print('Get training data positions...')
trainingDataPositions = {}
outer_index = 1
for word in words:
    if outer_index % 50 == 0:
        print(str(outer_index)+' / '+str(len(words)))
    outer_index += 1
    trainingDataPositions[word] = []
    for domain in domains:
        for i in range(len(data[domain])):
            if word in data[domain][i]:
                trainingDataPositions[word].append(i)
                break

print('Load all nearest neighbors...')
nearestNeighbors = []
names = []
for domain in domains:
    root = NYT_NEAREST_NEIGHBORS_FOLDER
    if domain == 'Europarl':
        root = EUROPARL_NEAREST_NEIGHBORS_FOLDER
    for dimension in [50,100,200,400,800]:
        #Load w2v
        data_temp = []
        for seed in [2518,2548,2590,29,401]:
            filename = root+'tenNearestNeighbors_'+domain+'_w2v_'+str(dimension)+'_'+str(seed)+'.pkl'
            if domain == 'Europarl':
                filename = root+'tenNearestNeighbors_w2v_'+str(dimension)+'_'+str(seed)+'.pkl'
            #print(filename)
            with open(filename,'rb') as pickledFile:
                data_temp.append(pickle.load(pickledFile))
        nearestNeighbors.append(data_temp)
        names.append(domain+'_w2v_'+str(dimension))
        
        #Load glove
        data_temp = []
        for seed in [2518,2548,2590,29,401]:
            filename = root+'tenNearestNeighbors_'+domain+'_glove_'+str(dimension)+'_'+str(seed)+'.pkl'
            if domain == 'Europarl':
                filename = root+'tenNearestNeighbors_glove_'+str(dimension)+'_'+str(seed)+'.pkl'
            #print(filename)
            with open(filename,'rb') as pickledFile:
                data_temp.append(pickle.load(pickledFile))
        nearestNeighbors.append(data_temp)
        names.append(domain+'_glove_'+str(dimension))

        #Load PPMI
        filename = root+'tenNearestNeighbors_'+domain+'_ppmi_'+str(dimension)+'.pkl'
        if domain == 'Europarl':
            filename = root+'tenNearestNeighbors_ppmi_'+str(dimension)+'.pkl'
        #print(filename)
        with open(filename,'rb') as pickledFile:
            nearestNeighbors.append([pickle.load(pickledFile)])
            names.append(domain+'_ppmi_'+str(dimension))

print('Start generating features...')
all_features = {}
for index in range(start_index,end_index):
    if index >= len(words):
        print('End of words array!')
        break

    word = words[index]
    print(str(index) + ' / ' + str(len(words)) + ' ' + word)
    word_features = wordFeatures(word,possible_pos,cmu_dict,wordDists)

    halt = False
    
    #pre-compute most-similar indices for each word
    similar = {}
    for nearestNeighbor,name in zip(nearestNeighbors,names):
        similar[name] = []
        for wordList in nearestNeighbor:
            similar[name].append(wordList[word])

    for domain1 in domains:
        for domain2 in domains:    
            corpus_features = corpusFeatures(trainingDataPositions,vocabOverlap,counters,domains,domain1,domain2,word,wordlists) #NOTE: This is using raw frequency, not normalized frequency

            for algorithm1 in ['w2v','glove','ppmi']:
                for algorithm2 in ['w2v','glove','ppmi']:
                    algorithm_features = algorithmFeatures(algorithm1,algorithm2)

                    for dimension1 in [50,100,200,400,800]:
                        for dimension2 in [50,100,200,400,800]:
                            name1 = domain1 + '_' + algorithm1 + '_' + str(dimension1)
                            name2 = domain2 + '_' + algorithm2 + '_' + str(dimension2)
                            algorithm_dimension_features = algorithmDimensionFeatures(dimension1,dimension2)

                            same = False
                            if name1==name2:
                                same = True
                            
                            target = stability(word,similar[name1],similar[name2],same=same)

                            features = word_features + corpus_features + algorithm_features + algorithm_dimension_features

                            all_features[(word,name1,name2)] = (features,target)

    if index > 0 and index % 500 == 0:
        print('saving')
        with open(OUTPUT_FOLDER+'all_features'+str(index-500)+'_'+str(index)+'.pkl','wb') as pickleFile:
            pickle.dump(all_features,pickleFile)
            all_features = {}

print('saving')
with open(OUTPUT_FOLDER+'all_features_end.pkl','wb') as pickleFile:
    pickle.dump(all_features,pickleFile)
