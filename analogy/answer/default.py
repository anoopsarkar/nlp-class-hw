import gensim.downloader as api
from gensim.models import KeyedVectors
import sys

def default(inputfile):
    model_gigaword = api.load("glove-wiki-gigaword-100")
    with open(inputfile) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line[0] == ':':
                print(line)
                continue
            (a, b, c) = line.split()
            results = model_gigaword.most_similar(positive=[a.lower(), c.lower()], negative=[b.lower()])
            print(results[0][0])

if __name__ == '__main__':
    import optparse
    import os
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    default(opts.input)

