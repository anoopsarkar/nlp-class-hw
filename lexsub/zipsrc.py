"""
Run:

    python3 zipsrc.py

This will create a file `source.zip` which you can upload to Coursys (courses.cs.sfu.ca) as your source code submission.

To customize the files used by default, run:

    python3 zipsrc.py -h
"""
import sys, os, optparse, shutil, iocollect

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-a", "--answerdir", dest="answer_dir", default='answer', help="answer directory containing your source files")
    optparser.add_option("-s", "--srcfile", dest="src_file", default='lexsub.py', help="name of source file for homework")
    optparser.add_option("-n", "--notebook", dest="notebook_file", default='lexsub.ipynb', help="name of iPython notebook for homework")
    optparser.add_option("-z", "--zipfile", dest="zipfile", default='source', help="zip file you should upload to Coursys (courses.cs.sfu.ca)")
    (opts, _) = optparser.parse_args()

    answer_files = iocollect.getfiles(opts.answer_dir)
    if opts.src_file not in answer_files:
        raise ValueError("Error: missing answer file {}. Did you name your answer program correctly?".format(opts.src_file))
    if opts.notebook_file not in answer_files:
        raise ValueError("Error: missing notebook file {}. Did you name your iPython notebook correctly?".format(opts.notebook_file))
    outputs_zipfile = shutil.make_archive(opts.zipfile, 'zip', opts.answer_dir)
    print("{0} created".format(outputs_zipfile), file=sys.stderr)

