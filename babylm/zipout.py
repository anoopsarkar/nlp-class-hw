"""
First make sure you have a Python3 program for your answer in ./answer/

Then run:

    python3 zipout.py

This will create a file `output.zip`.

To customize the files used by default, run:

    python3 zipout.py -h
"""

import sys, os, argparse, logging, tempfile, subprocess, shutil
import iocollect

class ZipOutput:

    def __init__(self, opts):
        self.input_dir = opts.input_dir # directory where input files are placed
        self.output_dir = opts.output_dir # directory for output files of your program

    def mkdirp(self, path):
        try:
            os.makedirs(path)
        except os.error:
            print("Warning: {} already exists. Existing files will be over-written.".format(path), file=sys.stderr)
            pass

    def run_all(self):
        self.mkdirp(self.output_dir)
        shutil.copytree(self.input_dir, self.output_dir, dirs_exist_ok=True)
        return True

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputdir", dest="input_dir", default=os.path.join('evaluation-pipeline', 'babylm', 'roberta-base-strict-small', 'zeroshot'), help="evaluation directory [default: evaluation-pipeline/babylm/roberta-base-strict-small/zeroshot/]")
    argparser.add_argument("-o", "--output", dest="output_dir", default='output', help="Save the output from the testcases to this directory.")
    argparser.add_argument("-z", "--zipfile", dest="zipfile", default='output', help="zip file with your output answers")
    argparser.add_argument("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    opts = argparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

    zo = ZipOutput(opts)
    if zo.run_all():
        outputs_zipfile = shutil.make_archive(opts.zipfile, 'zip', opts.output_dir)
        print("{} created".format(outputs_zipfile), file=sys.stderr)
    else:
        logging.error("problem in creating output zip file")
        sys.exit(1)
