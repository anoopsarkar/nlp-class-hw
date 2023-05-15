from collections import Counter

def accuracy(ref_data, output_data):
    (correct, total) = (0.0, 0.0)
    for (ref_sent, output_tsv) in zip(ref_data, output_data):
        (locations_str, output_sent) = output_tsv.split('\t')
        locations = [int(i) for i in locations_str.split(',')]
        ref = ref_sent.split()
        out = output_sent.split()
        for loc in locations:
            if ref[loc] == out[loc]:
                correct += 1
            total += 1
    return correct / total

