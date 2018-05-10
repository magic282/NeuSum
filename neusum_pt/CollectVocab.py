from __future__ import division
import operator

DefaultSpecialWords = ["<blank>", "<unk>", "<s>", "</s>"]


def Collect(inputFiles, vocabPath, toLower=False, userDefineSpecial=None):
    global DefaultSpecialWords
    specialWords = []
    if userDefineSpecial:
        for item in userDefineSpecial:
            if item not in specialWords:
                specialWords.append(item)
    else:
        specialWords = DefaultSpecialWords

    dict = CollectVocab(inputFiles, toLower)
    total = sum(dict.values())
    sorted_dict = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
    acc = 0
    with open(vocabPath, 'w', encoding='utf-8') as sw:
        count = 0
        for item in specialWords:
            sw.write("{0} {1}\n".format(item, count))
            count += 1
        for k, v in sorted_dict:
            if k in specialWords:
                continue
            acc += v
            sw.write("{0} {1} {2} {3}\n".format(k, count, v, 1.0 * acc / total))
            count += 1


def CollectVocab(files, toLower):
    dict = {}
    for f in files:

        with open(f, encoding='utf-8') as sr:
            for line in sr:
                line = line.strip()
                if toLower:
                    line = line.lower()
                sp = line.split()
                sp = filter(None, sp)
                for token in sp:
                    if token not in dict:
                        dict[token] = 0
                    dict[token] += 1
    return dict


if __name__ == "__main__":
    # files = [r"D:\users\v-qizhou\data\SongCi\v1\train\src.txt",
    #          r"D:\users\v-qizhou\data\SongCi\v1\train\tgt.txt"]
    # vocab_file = r"D:\users\v-qizhou\data\SongCi\v2\train\vocab.txt"  # TODO pay attention here
    # Collect(files, vocab_file, False, ["<blank>", "<unk>", "<s>", "</s>", "##SP##"])

    # files = [r"D:\users\v-qizhou\data\SongCi\v1\train\src.txt",
    #          r"D:\users\v-qizhou\data\SongCi\v1\train\tgt.txt"]
    # vocab_file = r"D:\users\v-qizhou\data\SongCi\v5\train\vocab.txt"  # TODO pay attention here
    # Collect(files, vocab_file, False, ["<blank>", "<unk>", "<s>", "</s>", "##SP##", "##KEYWORDS##", "##KEYSP##"])

    files = [r"D:\users\v-qizhou\data\SongCi\v7\train\tgt.txt"]
    vocab_file = r"D:\users\v-qizhou\data\SongCi\v7\train\vocab.txt"  # TODO pay attention here
    Collect(files, vocab_file, False, ["<blank>", "<unk>", "<s>", "</s>", "##SP##", "##NOTHING##", "##NONE##"])
