"""
First make sure you have a Python3 program for your answer in ./answer/

Then run:

    python3 zipout.py

This will create a file `output.zip`.

To customize the files used by default, run:

    python3 zipout.py -h
"""

import sys, os, optparse, logging, tempfile, subprocess, shutil
import iocollect

class ZipOutput:

    def __init__(self, opts):
        self.run_program = opts.run_program # solution to hw that is being tested
        self.python_bin = opts.python_bin # Python binary to run
        self.answer_dir = opts.answer_dir # name of directory where run_program exists
        self.input_dir = opts.input_dir # directory where input files  are placed
        self.output_dir = opts.output_dir # directory for output files of your program
        self.file_suffix = opts.file_suffix # file suffix for input files

    def mkdirp(self, path):
        try:
            os.makedirs(path)
        except os.error:
            print("Warning: {} already exists. Existing files will be over-written.".format(path), file=sys.stderr)
            pass

    def run(self, filename, path, output_path, base):
        """
        Runs a command specified by an argument vector (including the program name)
        and returns lists of lines from stdout and stderr.
        """

        # create the output files 
        if output_path is not None:
            stdout_path = os.path.join(output_path, "{}.out".format(base))
            stderr_path = os.path.join(output_path, "{}.err".format(base))

            # existing files are erased!
            stdout_file = open(stdout_path, 'w')
            stderr_file = open(stderr_path, 'w')
            status_path = os.path.join(output_path, "{}.ret".format(base))
        else:
            stdout_file, stdout_path = tempfile.mkstemp("stdout")
            stderr_file, stderr_path = tempfile.mkstemp("stderr")
            status_path = None

        run_program_path = os.path.abspath(os.path.join(self.answer_dir, self.run_program))
        run_python = os.path.abspath(self.python_bin)
        if os.path.exists(run_python) and os.access(run_python, os.X_OK):
            argv = [ run_python, run_program_path, '-i', filename ]
        else:
            print("Did not find {}. Are you sure you set up a virtualenv? Run `python3 -m venv venv` in the current directory.".format(self.python_bin), file=sys.stderr)
            if os.path.exists(self.run_program_path) and os.access(self.run_program_path, os.X_OK):
                argv = [ run_program_path, '-i', filename ]
            else:
                raise ValueError("Could not run {} {}".format(self.python_bin, self.run_program_path))

        stdin_file = open(filename, 'r')
        try:
            try:
                prog = subprocess.Popen(argv, stdin=stdin_file or subprocess.PIPE, stdout=stdout_file, stderr=stderr_file)
                if stdin_file is None:
                    prog.stdin.close()
                prog.wait()
            finally:
                if output_path is not None:
                    stdout_file.close()
                    stderr_file.close()
                else:
                    os.close(stdout_file)
                    os.close(stderr_file)
            if status_path is not None:
                with open(status_path, 'w') as status_file:
                  print(prog.returncode, file=status_file)
            with open(stdout_path) as stdout_input:
                stdout_lines = list(stdout_input)
            with open(stderr_path) as stderr_input:
                stderr_lines = list(stderr_input)
            if prog.stdin != None:
                prog.stdin.close()
            return stdout_lines, stderr_lines, prog.returncode
        except:
            print("error: something went wrong when trying to run the following command:", file=sys.stderr)
            print(argv, file=sys.stderr)
            raise
            #sys.exit(1)
        finally:
            if output_path is None:
                os.remove(stdout_path)
                os.remove(stderr_path)

    def run_path(self, path, files):
        # set up output directory
        if path is None or path == '':
            output_path = os.path.abspath(self.output_dir)
        else:
            output_path = os.path.abspath(os.path.join(self.output_dir, path))
        self.mkdirp(output_path)
        for filename in files:
            if path is None or path == '':
                testfile_path = os.path.abspath(os.path.join(self.input_dir, filename))
            else:
                testfile_path = os.path.abspath(os.path.join(self.input_dir, path, filename))
            if filename[-len(self.file_suffix):] == self.file_suffix:
                base = filename[:-len(self.file_suffix)]
                if os.path.exists(testfile_path):
                    print("running on input {}".format(testfile_path), file=sys.stderr)
                    self.run(testfile_path, path, output_path, base)

    def run_all(self):
        # check that a compiled binary exists to run on the input files
        argv = os.path.abspath(os.path.join(self.answer_dir, self.run_program))
        if not (os.path.isfile(argv)):
            logging.error("answer program missing: {}".format(argv))
            raise ValueError("Compile your source file to create an executable {}".format(argv))

        # check if input directory has subdirectories
        testcase_subdirs = iocollect.getdirs(os.path.abspath(self.input_dir))

        if len(testcase_subdirs) > 0:
            for subdir in testcase_subdirs:
                files = iocollect.getfiles(os.path.abspath(os.path.join(self.testcase_dir, subdir)))
                self.run_path(subdir, files)
        else:
            files = iocollect.getfiles(os.path.abspath(self.input_dir))
            self.run_path(None, files)

        return True

if __name__ == '__main__':
    #zipout_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    optparser = optparse.OptionParser()
    optparser.add_option("-r", "--run", dest="run_program", default='zhsegment.py', help="run this program against testcases [default: zhsegment.py]")
    optparser.add_option("-x", "--pythonbin", dest="python_bin", default='venv/bin/python3', help="run this binary of Python to run the program [default: python3]")
    optparser.add_option("-a", "--answerdir", dest="answer_dir", default='answer', help="answer directory [default: answer]")
    optparser.add_option("-i", "--inputdir", dest="input_dir", default=os.path.join('data', 'input'), help="testcases directory [default: data/input]")
    optparser.add_option("-e", "--ending", dest="file_suffix", default='.txt', help="suffix to use for testcases [default: .txt]")
    optparser.add_option("-o", "--output", dest="output_dir", default='output', help="Save the output from the testcases to this directory.")
    optparser.add_option("-z", "--zipfile", dest="zipfile", default='output', help="zip file with your output answers")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

    zo = ZipOutput(opts)
    if zo.run_all():
        outputs_zipfile = shutil.make_archive(opts.zipfile, 'zip', opts.output_dir)
        print("{} created".format(outputs_zipfile), file=sys.stderr)
    else:
        logging.error("problem in creating output zip file")
        sys.exit(1)

