
# Word Vectors and the Analogy Task

## Installation

Make sure you setup your virtual environment:

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

You can optionally copy and modify the requirements for when we
test your code:

    cp requirements.txt answer/requirements.txt

## Required files

You must create the following files:

    answer/analogy.py
    answer/analogy.ipynb

You can copy over the `default.py` and `default.ipynb` files to start off (see below).

## Create output.zip

To create the `output.zip` file for upload to Coursys do:

    python zipout.py

For more options:

    python zipout.py -h

## Create source.zip

To create the `source.zip` file for upload to Coursys do:

    python zipsrc.py

For more options:

    python zipsrc.py -h

## Check your accuracy

To check your accuracy on the dev set:

    python check.py

For more options:

    python check.py -h

In particular use the log file to check your output evaluation:

    python check.py -l log

The accuracy on `data/input/test.txt` will not be shown.  We will
evaluate your output on the test input after the submission deadline.

## Default solution

The default solution is provided in `default.py`. To use the default
as your solution:

    cp default.py answer/lexsub.py
    cp default.ipynb answer/lexsub.ipynb
    python zipout.py
    python check.py

Make sure that the command line options are kept as they are in
`default.py`. You can add to them but you must not delete any
command line options that exist in `default.py`.

Submitting the default solution without modification will get you
zero marks.

## Data files

The data files provided are:

* `data/sample_vec.txt` -- small sample word vector file
* `data/lexicons` -- different lexicons / ontologies used for retrofitting
* `data/input` -- input files `dev.txt` and `test.txt`
* `data/reference/dev.out` -- reference output for the `dev.txt` input file
* `data/train` -- training data to create your own lexicon files.


## Downloading from gensim

To get the word2vec model as a text file:

    python -m gensim.downloader --download glove-wiki-gigaword-100
