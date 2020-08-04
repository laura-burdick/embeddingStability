# Factors Influencing the Surprising Instability of Word Embeddings
Laura (Wendlandt) Burdick, Jonathan K. Kummerfeld, Rada Mihalcea

Language and Information Technologies (LIT)

University of Michigan

## Introduction
The code in this repository was used in the paper "Factors Influencing the Surprising Instability of Word Embeddings" by Wendlandt, et al. I have tried to document it well, but at the end of the day, it is research code, so if you have any problems using it, please get in touch with Laura Burdick (lburdick@umich.edu).

## Citation Information
If you use this code, please cite the following paper:
```
@inproceedings{Wendlandt18Surprising,
author = {Wendlandt, Laura and Kummerfeld, Jonathan K. and Mihalcea, Rada},
title = {Factors Influencing the Surprising Instability of Word Embeddings},
pages = "2092--2102",
url = "https://www.aclweb.org/anthology/N18-1190",
doi = "10.18653/v1/N18-1190",
booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies",
year = {2018}
}
```

## Code Included
**stability.py**: Includes functions for calculating the stability of a word (as well as calculating the most similar words to that word, which is needed for calculating stability)

**example.py**: A toy example of how to use the stability code

**Regression Model**: Code to replicate the regression model found in the paper, as well as a pre-trained model (Quick Start guide [here](regression/README.md))

## Acknowledgements
We would like to thank Ben King and David Jurgens for helpful discussions about this paper, as well as our anonymous reviewers for useful feedback. This material is based in part upon work supported by the National Science Foundation (NSF \#1344257) and the Michigan Institute for Data Science (MIDAS). Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the NSF or MIDAS. 
