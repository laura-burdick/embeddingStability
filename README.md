Factors Influencing the Surprising Instability of Word Embeddings
=================================================================

The code in this repository was used in the paper "Factors Influencing the Surprising Instability of Word Embeddings" by Wendlandt, et al. I have tried to document it well, but at the end of the day, it is research code, so if you have any problems using it, please get in touch with Laura Wendlandt (wenlaura@umich.edu).

If you use this code, please cite the following paper:
```
@article{Wendlandt18Surprising,
author = {Wendlandt, L. and J. Kummerfeld and R. Mihalcea},
title = {Factors Influencing the Surprising Instability of Word Embeddings},
journal = {NAACL-HLT},
year = {2018}
}
```

REGRESSION MODEL
================
The following files and instructions can be used to reproduce the regression model described in the paper.

1. Pre-process data
	- Download The New York Times Annotated Corpus from LDC (https://catalog.ldc.upenn.edu/ldc2008t19)
	- Download all of English Europarl (http://www.statmt.org/europarl/)
	- To pre-process data: lowercase data, replace all digits with hash tags, word_tokenize (using NLTK word_tokenize()), and sentence tokenize (using NLTK sent_tokenize()) 
	- For each dataset file, run getWordList.py to generate a saved word list

2. Train embedding spaces
	- Run w2v_europarl.py and w2v_nyt.py to train w2v embedding spaces
	- Run ppmi_europarl.py and ppmi_nyt.py to train PPMI embedding spaces
	- GloVe models were trained using original GloVe code (https://nlp.stanford.edu/projects/glove/), with a slight modification. The code was modified to take a random seed as an argument (used to seed the random number generator). Each model was then saved in the save format as the word2vec and PPMI models.

3. Build balltrees for all embedding spaces (to speed up all nearest neighbor computation at a later step)
	- Run create_balltrees_nyt.py and create_balltrees_europarl.py
