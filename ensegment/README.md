
# Homework 0

## Installation

Make sure you setup your virtual environment:

    python3 -m venv venv
    source venv/bin/activate

## Required files

You must create the following files:

    answer/ensegment.py
    answer/ensegment.ipynb

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

The accuracy on `data/input/test.txt` will not be shown.
We will evaluate your output on the test input after the
submission deadline.

## Default solution

The default solution is provided in `default.py`. To use the default as your solution:

    cp default.py answer/ensegment.py
    cp default.ipynb answer/ensegment.ipynb
    python3 zipout.py
    python3 check.py

Note that the default solution will get you zero marks.
