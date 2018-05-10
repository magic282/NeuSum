import sys
from PyRouge.Rouge import Rouge

rouge = Rouge.Rouge()


def evaluate(ref_file, pred_file):
    print(ref_file)
    print(pred_file)

    all_pred = []
    all_ref = []

    with open(pred_file, 'r', encoding='utf-8') as pred_reader, \
            open(ref_file, 'r', encoding='utf-8') as ref_reader:
        for src_line, tgt_line in zip(pred_reader, ref_reader):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()
            sp = src_line.split("\t")
            pred = sp[1]
            tgt_sents = tgt_line.split("##SENT##")
            all_pred.append(pred)
            all_ref.append(' '.join(tgt_sents))

    score = rouge.compute_rouge(all_ref, all_pred)
    print(score)


def main():
    pred_file = r'foo'
    ref_file = r'bar'
    raise Exception('')
    evaluate(ref_file, pred_file)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        evaluate(sys.argv[1], sys.argv[2])
    else:
        main()
