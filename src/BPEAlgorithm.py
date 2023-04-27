import re
from collections import Counter, defaultdict

def build_vocab(corpus: str) -> dict:

    """ Step 1. Build vocab from text corpus """
    # separate each char in word by space and add mark end of token
    tokens =[" ".join(word) + "</w>" for word in corpus.split()]

    #count frequency of tokens in corpus

    vocab = Counter(tokens)

    return vocab


def get_stats(vocab: dict) -> dict:

    """ Step 2. Get counts of pairs of consecutive symbols """

    pairs = defaultdict(int)

    for word, frequency in vocab.items():
        symbols = word.split()

        # counting up occurrences of pairs

        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += frequency
    
    return pairs


def merge_vocab(pair: tuple, v_in: dict) -> dict:

    """ Step 3: Merge all occurences of the most frequent pair """
    v_out = {}

    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S) ')

    for word in v_in:

        # replace most frequent pair in all vocabulary

        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    
    return v_out

corpus = 'sot eshte nje dite e bukur. neser do te jete nje dite e bukur. neser do te jete nje dite e keqe. neser do te jete nje dite e shemtuar'
vocab = build_vocab(corpus)
num_merges = 50

for i in range(num_merges):

    pairs = get_stats(vocab)

    if not pairs:
        break

    best = max(pairs, key = pairs.get)
    vocab = merge_vocab(best, vocab)

print(vocab)




