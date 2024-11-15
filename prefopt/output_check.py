from collections import Counter
import sys
import logging
import json

def precision(ref_data, output_data):
    tp = 0.
    fp = 0.
    for (ref, output) in zip(ref_data, output_data):
        ref_dict  = json.loads(ref)
        output_dict = json.loads(output)
        ref_str = ref_dict['output']
        output_str = output_dict['output']
        if ref_str.split()[0].lower() == output_str.split()[0].lower():
            tp += 1.
            logging.info(f"CORRECT:ref={ref.lower()} output={output}")
        else:
            fp += 1.
            logging.info(f"INCORRECT:ref={ref.lower()} output={output}")
    return (tp / (tp + fp))

if __name__ == '__main__':
    import os, optparse
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--refcases", dest="ref", default=os.path.join('data', 'reference', 'dev.out'), help="references [default: data/reference/dev.out]")
    optparser.add_option("-o", "--outputfile", dest="output", default='output.txt', help="output file created by prefopt.py [default: output.txt]")
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
