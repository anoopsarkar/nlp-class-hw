# Evaluation pipeline

Clone the official `evaluation-pipeline` repository for the BabyLM
challenge inside this directory.

    git clone https://github.com/babylm/evaluation-pipeline
    cd evaluation-pipeline
    unzip filter_data.zip

On line 49 of the file `babylm_eval.py` change from `cuda` to `cpu`
if you are going to run the evaluation without using a GPU.

A pre-trained model trained on the `babylm_10M` data (see above for
details) has been made available for you so you don't need to train
a new model from scratch on the 10M data. Follow the link below to
download the model:

    https://drive.google.com/drive/folders/1M85dyfSngIrChY4u-w2jCtSxC60mHZJz?usp=sharing

When downloaded keep the directory `roberta-base-strict-small`
inside the `evaluation-pipeline` directory where you checked out
the BabyLM evaluation pipeline repository.

You can run the unmodified pre-trained model through the evaluation pipeline as follows:

    python babylm_eval.py "roberta-base-strict-small" "encoder" > babylm_eval.output

This will produce scores on the BLiMP task:

    Scores:
    anaphor_agreement:  81.54%
    argument_structure: 67.12%
    binding:    67.26%
    control_raising:    67.85%
    determiner_noun_agreement:  90.75%
    ellipsis:   76.44%
    filler_gap: 63.48%
    irregular_forms:    87.43%
    island_effects:     39.87%
    npi_licensing:      55.92%
    quantifiers:        70.53%
    subject_verb_agreement:     65.42%

It will also create the directories and files in:
`babylm/roberta-base-strict-small/zeroshot/` for each of the BLiMP subtasks.

Make sure you have the following directory contents before running `zipout.py`:

    ls evaluation-pipeline/babylm/roberta-base-strict-small/zeroshot/

You still have `zipout.py` and `check.py` to submit your final
evaluation scores on the zero-shot task for the BabyLM challenge,
which is the BLiMP evaluation.

    python3 zipout.py
    python3 check.py

For this homework, it is acceptable to submit the evaluation scores
produced by the provided pre-trained model, since this homework is
to evaluate if you can get started on a project and run the evaluation
pipeline to evaluate your project work on standard benchmark
dataset(s).

## Submit your homework on Coursys

Once you are done with your homework submit all the relevant materials
to Coursys for evaluation.

### Create output.zip

Once you have run the BabyLM evaluation pipeline create `output.zip`
for upload to Coursys using:

    python3 zipout.py

### Create source.zip

To create the `source.zip` file for upload to Coursys do:

    python3 zipsrc.py

You must have the following files or `zipsrc.py` will complain about it:

* `answer/babylm.py` -- this can be an empty file for this homework if you didn't change the pre-trained model.
* `answer/babylm.ipynb` -- this is the iPython notebook that will be your write-up for the homework.

Each group member should write about what they did for this homework in the Python notebook.

