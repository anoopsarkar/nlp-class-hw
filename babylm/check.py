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

import sys, os, argparse, logging, tempfile, subprocess, shutil, difflib, io, json
from collections import defaultdict
import iocollect

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-z", "--zipfile", dest="zipfile", default='output.zip', help="zip file created by zipout.py [default: output.zip]")
    argparser.add_argument("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    opts = argparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

    try:
        with open(opts.zipfile, 'rb') as f:
            zipfile = io.BytesIO(f.read())
            zip_data = iocollect.extract_zip(zipfile) # contents of output zipfile produced by `python zipout.py` as a dict
            tally = 0.0
            num_keys = 0
            for key in zip_data:
                if key.endswith('eval_results.json'):
                    num_keys += 1
                    json_data = json.loads(zip_data[key])
                    tally += json_data['eval_accuracy']
                    logging.info(f"{key} accuracy = {json_data['eval_accuracy']}")
            score = (tally / num_keys) * 100.0
            print(f"score: {score:.4f}")
    except Exception as e:
        print("Could not process zipfile: {}".format(opts.zipfile), file=sys.stderr)
        print("ERROR: {}".format(str(e)), file=sys.stderr)
