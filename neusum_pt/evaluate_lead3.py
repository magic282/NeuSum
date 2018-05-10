import sys
from PyRouge.Rouge import Rouge

rouge = Rouge.Rouge()


def main(src_file, tgt_file):
    all_lead3 = []
    all_ref = []

    with open(src_file, 'r', encoding='utf-8') as src_reader, \
            open(tgt_file, 'r', encoding='utf-8') as tgt_reader:
        for src_line, tgt_line in zip(src_reader, tgt_reader):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()
            src_sents = src_line.split("##SENT##")
            tgt_sents = tgt_line.split("##SENT##")
            lead3 = src_sents[:3]
            all_lead3.append(' '.join(lead3))
            all_ref.append(' '.join(tgt_sents))
            # all_lead3.append(lead3)
            # all_ref.append(tgt_sents)

    score = rouge.compute_rouge(all_ref, all_lead3)
    print(score)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
