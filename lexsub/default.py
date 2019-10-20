import os, sys, optparse
import tqdm
import pymagnitude

class LexSub:

    def __init__(self, wvec_file, topn=10):
        self.wvecs = pymagnitude.Magnitude(wvec_file)
        self.topn = topn

    def substitutes(self, index, sentence):
        "Return ten guesses that are appropriate lexical substitutions for the word at sentence[index]."
        return(list(map(lambda k: k[0], self.wvecs.most_similar(sentence[index], topn=self.topn))))

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="input file with target word in context")
    optparser.add_option("-w", "--wordvecfile", dest="wordvecfile", default=os.path.join('data', 'glove.6B.100d.magnitude'), help="word vectors file")
    optparser.add_option("-n", "--topn", dest="topn", default=10, help="produce these many guesses")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    lexsub = LexSub(opts.wordvecfile, int(opts.topn))
    num_lines = sum(1 for line in open(opts.input,'r'))
    with open(opts.input) as f:
        for line in tqdm.tqdm(f, total=num_lines):
            fields = line.strip().split('\t')
            print(" ".join(lexsub.substitutes(int(fields[0].strip()), fields[1].strip().split())))
