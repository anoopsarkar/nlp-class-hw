
# Advanced NLP: Homework 1

## Installation

Make sure you setup your virtual environment:

    python3.10 -m venv venv
    source venv/bin/activate
    pip install -U -r requirements.txt

You can optionally copy and modify the requirements for when we
test your code:

    cp requirements.txt answer/requirements.txt

## Required files

You must create the following files:

    answer/prefixtune.py
    answer/prefixtune.ipynb

## Create output.zip

To create the `output.zip` file for upload to Coursys do:

    python3 zipout.py

For more options:

    python3 zipout.py -h

## Create source.zip

To create the `source.zip` file for upload to Coursys do:

    python3 zipsrc.py

For more options:

    python3 zipsrc.py -h

## Check your accuracy

To check your accuracy on the dev set:

    python3 check.py

For more options:

    python3 check.py -h

In particular use the log file to check your output evaluation:

    python3 check.py -l log

The accuracy on `data/input/test.txt` will not be shown.  We will
evaluate your output on the test input after the submission deadline.

## Default solution

The default solution is provided in `default.py`. To use the default
as your solution:

    cp default.py answer/prefixtune.py
    cp default.ipynb answer/prefixtune.ipynb
    python3 zipout.py
    python3 check.py

Make sure that the command line options are kept as they are in
`default.py`. You can add to them but you must not delete any
command line options that exist in `default.py`.

Submitting the default solution without modification will get you
zero marks.

## Default solution model file

The default model file is the `distilgpt2` model file that will be
downloaded when you run `default.py` for the first time.

## Data files

The data files provided are:

* `data/train.txt.gz` -- the training data used to train the `answer/default.py` model (`default.py` uses the [huggingface dataset for E2E](https://huggingface.co/datasets/e2e_nlg) to load the data but a copy of the training data is provided just in case).
* `data/input` -- input files `dev.txt` and `test.txt` infected with noise. a subset of `dev.txt` is provided as `small.txt` for development of your solution.
* `data/reference/dev.out` -- the reference output for the `dev.txt` input file
* `data/reference/small.out` -- the reference output for the `dev.txt` input file
