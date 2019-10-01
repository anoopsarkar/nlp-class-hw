from collections import Counter

def fscore(ref_data, output_data):
    (tp, fp, fn) = (0, 0, 0) # true positives, false positives, false negatives
    for (ref_sent, output_sent) in zip(ref_data, output_data):
        ref_words = Counter(ref_sent.split())
        output_words = Counter(output_sent.split())
        output_diff = output_words
        output_diff.subtract(ref_words)
        tp += len([ output_diff[x] for x in output_diff if output_diff[x] == 0 ])
        fp += len([ output_diff[x] for x in output_diff if output_diff[x] > 0 ])
        fn += len([ output_diff[x] for x in output_diff if output_diff[x] < 0 ])
    if (tp + fp == 0) or (tp + fn == 0):
        return 0
    # compute precision and recall and return f-score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return (2 * ((precision * recall) / (precision + recall)))

if __name__ == '__main__':
    import os, optparse
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--refcases", dest="ref", default=os.path.join('data', 'reference', 'dev.out'), help="references [default: data/reference/dev.out]")
    optparser.add_option("-o", "--outputfile", dest="output", default='output.txt', help="output file created by zhsegment.py [default: output.txt]")
    (opts, _) = optparser.parse_args()

    with open(opts.ref, 'rb') as refh:
        ref_data = [str(x).strip() for x in refh.read().splitlines()]
    with open(opts.output, 'rb') as outh:
        out_data = [str(x).strip() for x in outh.read().splitlines()]
        output_data = out_data[:len(ref_data)]
        if len(ref_data) == len(output_data):
            print("Fscore=", fscore(ref_data, output_data))
        else:
            raise ValueError("reference and output are different lengths")

