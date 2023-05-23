"""
Run:

    python3 zipsrc.py

This will create a file `source.zip` which you can upload to Coursys (courses.cs.sfu.ca) as your source code submission.

To customize the files used by default, run:

    python3 zipsrc.py -h
"""
import sys, os, argparse, shutil, iocollect

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-a", "--answerdir", dest="answer_dir", default='answer', help="answer directory containing your source files")
    argparser.add_argument("-s", "--srcfile", dest="src_file", default='bertchunker.py', help="name of source file for homework")
    argparser.add_argument("-n", "--notebook", dest="notebook_file", default='bertchunker.ipynb', help="name of iPython notebook for homework")
    argparser.add_argument("-z", "--zipfile", dest="zipfile", default='source', help="zip file you should upload to Coursys (courses.cs.sfu.ca)")
    opts = argparser.parse_args()
    answer_files = iocollect.getfiles(opts.answer_dir)
    if opts.src_file not in answer_files:
        raise ValueError("Error: missing answer file {}. Did you name your answer program correctly?".format(opts.src_file))
    if opts.notebook_file not in answer_files:
        raise ValueError("Error: missing notebook file {}. Did you name your iPython notebook correctly?".format(opts.notebook_file))
    outputs_zipfile = shutil.make_archive(opts.zipfile, 'zip', opts.answer_dir)
    print("{0} created".format(outputs_zipfile), file=sys.stderr)
