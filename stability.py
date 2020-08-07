# Author: Laura Burdick (lburdick@umich.edu)
# Dependencies: sklearn, faiss, numpy, tqdm
# Code to find the most similar words to a given word in an embedding space, as well as code to calculate stability

import faiss
import pickle
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm,trange

# Function to find the most similar words in an embedding space for all relevant words
# Similarity is measured using cos similarity
#
# @param embeddings (list of numpy.ndarray)
#	The embedding space to search for similar words in.
#	The embedding space should be in list format, where the list is a list of embeddings, where each embedding is an numpy.ndarray (each embedding corresponds to a word in the words list)
# @param embedding_words (list of str)
#	List of words to find similar words for
# @param topn (int)
#	The number of similar words to find
#
# @returns nearestNeighbors (dict)
#	The topn most similar words, sorted by cos similarity.
#	Formatted as a dictionary, where the keys are the words, and the values are lists of nearest words.
#
def mostSimilar(embeddings,embedding_words,topn=10):	
	xb = np.matrix([[float(j) for j in i[1:]] for i in embeddings],dtype='float32') #database

	print('Normalizing vectors')
	for i in trange(len(xb)):
		xb[i] = normalize(xb[i])

	d = xb.shape[1] #dimension
	nb = xb.shape[0] #database size
	nq = len(embedding_words) #num queries
	print('d',d)
	print('nb',nb)
	print('nq',nq)

	print('Creating query matrix...')
	xq = xb[[i for i in range(len(embedding_words))],:]
	print(xq.shape)

	print('Building index...')
	faiss_index = faiss.IndexFlatL2(d)
	faiss_index.add(xb) 

	k = topn+1 #number of nearest neighbors

	print('Calculating nearest neighbors...')
	D, I = faiss_index.search(xq, k)
	
	nearestNeighbors = {}
	print('Recording nearest neighbors...')
	for i in tqdm(range(len(embedding_words))):
		word = embedding_words[i]
		nearestNeighbors[word] = [embedding_words[j] for j in I[i]][1:]
	
	return nearestNeighbors

# Calculates the stability of a word in two sets of embedding spaces
# Assumes that you've already calculated the most similar words for the word
#
# @param word
#    The word to calculate stability for
# @param similar1
#    The list of nearest neighbors to word in the first set of embedding spaces
#    len(similar1) = # of embedding spaces in the first set
#    For each i, len(similar1[i]) = # of nearest neighbors to consider (same for each i)
# @param similar2
#    The list of nearest neighbors to word in the second set of embedding spaces
# @param same
#    Are the two lists of embedding spaces the same? (default = False)
#
# @returns a float, the average stability of the word across the two sets of spaces
#
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
