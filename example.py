# Author: Laura Wendlandt
# Dependencies: nltk, gensim
# Example: Calculating stability for a word across two word2vec spaces

from stability import mostSimilar,stability
import nltk
from nltk.corpus import brown
from gensim.models import word2vec

# Download and format corpora
nltk.download('brown')

sentences = brown.sents()
sentences = [[word.lower() for word in sentence] for sentence in sentences]

# List of words that we want stability for
embedding_words = ['president','country','the','government','congress']

print("Training two word2vec models on the Brown corpus")
model1 = word2vec.Word2Vec(sentences,size=100,window=5,min_count=1,seed=42)
model2 = word2vec.Word2Vec(sentences,size=100,window=5,min_count=1,seed=102)

print('Formatting embedding spaces for stability code')
embeddings1 = [model1.wv[word] for word in embedding_words]
embeddings2 = [model2.wv[word] for word in embedding_words]

print('Calculating stability')
mostSimilar1 = mostSimilar(embeddings1,embedding_words)
mostSimilar2 = mostSimilar(embeddings1,embedding_words)

for word in embedding_words:
	stab = stability(word,[mostSimilar1[word],mostSimilar2[word]],[mostSimilar1[word],mostSimilar2[word]],True)
	print(word,'has a stability of ' + str(stab*10) + '%')
