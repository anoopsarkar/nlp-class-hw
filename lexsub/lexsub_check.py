from collections import Counter

def precision(ref_data, output_data):
    tp = 0.
    fp = 0.
    for (ref, output) in zip(ref_data, output_data):
        truth = set(ref.split('\t')[1].split())
        output_candidates = set(output.split()[:10])
        if truth & output_candidates:
            tp += 1.
        else:
            fp += 1.
    return (tp / (tp + fp))

if __name__ == '__main__':
    import os, optparse
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--refcases", dest="ref", default=os.path.join('data', 'reference', 'dev.out'), help="references [default: data/reference/dev.out]")
    optparser.add_option("-o", "--outputfile", dest="output", default='output.txt', help="output file created by lexsub.py [default: output.txt]")
    (opts, _) = optparser.parse_args()

    with open(opts.ref, 'rt') as refh:
        ref_data = [str(x).strip() for x in refh.read().splitlines()]
    with open(opts.output, 'rt') as outh:
        out_data = [str(x).strip() for x in outh.read().splitlines()]
        output_data = out_data[:len(ref_data)]
        if len(ref_data) == len(output_data):
            print("Score={:.4f}".format(100*precision(ref_data, output_data)))
        else:
            raise ValueError("reference and output are different lengths")
