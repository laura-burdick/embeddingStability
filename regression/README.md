# Regression Model

## Dependencies
- Python 3 (NOTE: doesn't work with Python 2.7!!!!)
- NLTK (https://www.nltk.org/)
- Gensim (https://radimrehurek.com/gensim/)
- Python Numpy (http://www.numpy.org/)
- Python Scipy (https://www.scipy.org/)
- Python Scikit-Learn (http://scikit-learn.org/stable/)
- Python TQDM (https://pypi.python.org/pypi/tqdm)

## Quick Start
The following files and instructions can be used to reproduce the regression model described in the paper. (Please NOTE that all of this code needs to be run with Python 3, not Python 2.7!)

1. Pre-process data
	- Download The New York Times Annotated Corpus from LDC (https://catalog.ldc.upenn.edu/ldc2008t19)
	- Download all of English Europarl (http://www.statmt.org/europarl/)
	- To pre-process data: lowercase data, replace all digits with hash tags, word_tokenize (using NLTK word_tokenize()), and sentence tokenize (using NLTK sent_tokenize()) 
	- For each domain file, run getWordList.py to generate a saved word list (e.g., run this for Europarl and for each NYT domain)

2. Train embedding spaces
	- Run w2v_europarl.py and w2v_nyt.py to train w2v embedding spaces
	- Run ppmi_europarl.py and ppmi_nyt.py to train PPMI embedding spaces
	- GloVe models were trained using original GloVe code (https://nlp.stanford.edu/projects/glove/), with a slight modification. The code was modified to take a random seed as an argument (used to seed the random number generator). Each model was then saved in the same format as the word2vec and PPMI models.

3. Build balltrees for all embedding spaces (to speed up all nearest neighbor computation at a later step)
	- Run create_balltrees_nyt.py and create_balltrees_europarl.py

4. Pre-calculate ten nearest neighbors for each word in each embedding space
	- Run precalculateNearestNeighbors_nyt.py and precalculateNearestNeighbors_europarl.py

5. Get shared word list
	- Run sharedWordList.py

6. Generate all classifier features
	- Run generateClassifierFeatures.py

7. Train regression model
	- Run regressionModel.py
