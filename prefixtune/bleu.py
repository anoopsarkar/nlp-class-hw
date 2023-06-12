import sys, re
import sacrebleu

bleu = sacrebleu.metrics.BLEU(effective_order=True)

def compute_bleu(references, output_data):
    bleu_score = 0.0
    if len(references) == len(output_data):
        score = 0.0
        total = 0.0
        for line in output_data:
            r = references[line[0]]
            h = line[1]
            score += bleu.sentence_score(h, r).score
            total += 1.
        bleu_score = score / total
    return bleu_score

if __name__ == '__main__':
    import os, argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-t", "--refcases", dest="ref", default=os.path.join('data', 'reference', 'dev.out'), help="references [default: data/reference/dev.out]")
    argparser.add_argument("-o", "--outputfile", dest="output", default='output.txt', help="output file created by chunker.py [default: output.txt]")
    opts = argparser.parse_args()

    references = {}
    ref_data = []
    output_data = []
    with open(opts.ref, 'r') as ref:
        ref_data = list(filter(lambda k: k, [str(x) for x in ref.read().splitlines()]))
        for line in ref_data:
            src_id, _, suggested_reference = line.split('||')
            references.setdefault(src_id, [])
            references[src_id].append(suggested_reference)
    with open(opts.output) as out:
        output_data = list(filter(lambda k: k, [str(x) for x in out.read().splitlines()]))
        output_data = [line.split('||') for line in output_data]
        output_data = output_data[:len(ref_data)]
    print(f"bleu score: {compute_bleu(references, output_data)}")
