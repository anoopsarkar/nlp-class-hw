import sacrebleu
import sys

def bleu(ref_t, pred_t):
    return sacrebleu.corpus_bleu(pred_t, [ref_t], force=True, lowercase=True, tokenize='none')

if __name__ == '__main__':
    import os, optparse
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--refcases", dest="ref", default=os.path.join('data', 'reference', 'dev.out'), help="references [default: data/reference/dev.out]")
    optparser.add_option("-o", "--outputfile", dest="output", default='output.txt', help="output file created by attention.py [default: output.txt]")
    (opts, _) = optparser.parse_args()

    with open(opts.ref, 'r') as refh:
        ref_data = [str(x).strip() for x in refh.read().splitlines()]
    with open(opts.output, 'r') as outh:
        out_data = [str(x).strip() for x in outh.read().splitlines()]
        output_data = out_data[:len(ref_data)]
        if len(ref_data) == len(output_data):
            print(bleu(ref_data, output_data))
        else:
            raise ValueError("reference and output are different lengths")

