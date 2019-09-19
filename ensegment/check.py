"""
First run `python3 zipout.py` to create your output zipfile `output.zip` and output directory `./output`

Then run:

    python3 check.py

It will print out a score of all your outputs that matched the
testcases with a reference output file (typically `./references/dev/*.out`).
In some cases the output is supposed to fail in which case it
compares the `*.ret` files instead.

To customize the files used by default, run:

    python3 check.py -h
"""

import sys, os, optparse, logging, tempfile, subprocess, shutil, difflib, io
from collections import defaultdict
import iocollect
import ensegment_check

class Check:

    def __init__(self, ref_dir):
        self.ref_dir = ref_dir                  # directory where references are placed
        self.linesep = "{}".format(os.linesep) # os independent line separator
        self.path_score = {'dev.out': 1, 'test.out': 1} # set up this dict to score different testcases differently
        self.default_score = 1                  # default score if it does not exist in path_values

        # perf is a dict used to keep track of total score based on testcase type with three keys:
        # each element of perf is a dict with three (key, value) pairs
        # num_correct: used to keep track of how many were correctly matched to reference output
        # total: used to keep track of total number of testcases with reference outputs
        # score: total score earned which depends on self.path_score or default_score
        self.perf = {}

    def check_path(self, path, files, zip_data):
        logging.info("path: {}".format(path))
        logging.info("files: {}".format(files))
        testfile_path = ''
        testfile_key = ''
        path_key = ''
        for filename in files:
            logging.info("testing filename: {}".format(filename))
            if path is None or path == '':
                logging.info("filename={}".format(filename))
                testfile_path = os.path.abspath(os.path.join(self.ref_dir, filename))
                testfile_key = filename
                path_key = filename
            else:
                logging.info("path={}".format(path))
                testfile_path = os.path.abspath(os.path.join(self.ref_dir, path, filename))
                testfile_key = os.path.join(path, filename)
                path_key = path

            logging.info("path_key={}".format(path_key))

            # set up score value for matching output correctly
            score = self.default_score
            if path_key in self.path_score:
                score = self.path_score[path_key]
            tally = 0.0
            self.perf[path_key] = 0.0

            logging.info("Checking {}".format(testfile_key))
            if testfile_key in zip_data:
                with open(testfile_path, 'rb') as ref:
                    ref_data = [str(x).strip() for x in ref.read().splitlines()]
                    output_data = [str(x).strip() for x in zip_data[testfile_key].splitlines()]
                    output_data = output_data[:len(ref_data)]
                    if len(ref_data) == len(output_data):
                        logging.info("ref, output {}".format(list(zip(ref_data, output_data))))
                        tally = ensegment_check.fscore(ref_data, output_data)
                        logging.info("score {}: {}".format(testfile_key, tally))
                    else:
                        logging.info("length mismatch between output and reference")
                        tally = 0.

            self.perf[path_key] = tally

    def check_all(self, zipcontents):
        zipfile = io.BytesIO(zipcontents)
        zip_data = iocollect.extract_zip(zipfile) # contents of output zipfile produced by `python zipout.py` as a dict

        # check if references has subdirectories
        ref_subdirs = iocollect.getdirs(os.path.abspath(self.ref_dir))
        if len(ref_subdirs) > 0:
            for subdir in ref_subdirs:
                files = iocollect.getfiles(os.path.abspath(os.path.join(self.ref_dir, subdir)))
                self.check_path(subdir, files, zip_data)
        else:
            files = iocollect.getfiles(os.path.abspath(self.ref_dir))
            self.check_path(None, files, zip_data)
        return self.perf

if __name__ == '__main__':
    #check_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--refcases", dest="ref_dir", default=os.path.join('data', 'reference'), help="references directory [default: data/reference]")
    optparser.add_option("-z", "--zipfile", dest="zipfile", default='output.zip', help="zip file created by zipout.py [default: output.zip]")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

    check = Check(ref_dir=opts.ref_dir)
    try:
        with open(opts.zipfile, 'rb') as f:
            perf = check.check_all(f.read())
            if perf is not None:
                total = 0
                for (d, tally) in perf.items():
                    print("{0} score: {1:.2f}".format(d, tally))
            else:
                print("Nothing to report!")
    except Exception as e:
        print("Could not process zipfile: {}".format(opts.zipfile), file=sys.stderr)
        print("ERROR: {}".format(str(e)), file=sys.stderr)

