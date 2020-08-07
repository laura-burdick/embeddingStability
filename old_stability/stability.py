# Author: Laura Wendlandt
# Dependencies: sklearn
# Code to find the most similar words to a given word in an embedding space, as well as code to calculate stability

from sklearn.metrics.pairwise import cosine_similarity

# Function to find the most similar words in an embedding space to a particular word.
# Similarity is measured using cos similarity
# NOTE: if you need to speed this up, consider using balltrees
#
# @param embeddingSpace (dict)
#     The embedding space to search for similar words in.
#     The embedding space should be in dict format, where key is word and value is embedding
# @param word (str)
#     The word to find most similar words in
# @param topn (int)
#     The number of similar words to find
#
# @returns mostSimilar (list of strs)
#     The topn most similar words, sorted by cos similarity
#
def mostSimilar(embeddingSpace,word,topn=10):
    wordRep = embeddingSpace[word].reshape(1,-1)
    words = []
    sims = []
    for key,value in embeddingSpace.items():
        if key == word:
            continue
        words.append(key)
        sims.append(cosine_similarity(wordRep,value.reshape(1,-1)))
    sortedList = [list(x) for x in zip(*sorted(zip(words, sims), key=lambda pair: pair[1], reverse=True))]
    return sortedList[0][:topn]

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
