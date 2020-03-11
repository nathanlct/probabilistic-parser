from nltk.grammar import Nonterminal, induce_pcfg
from nltk.tree import Tree
from nltk.treetransforms import un_chomsky_normal_form

import time
from collections import defaultdict
import numpy as np
import os
import pickle
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cosine

from utils import *


DEBUG = False
BEAM_SIZE = 50  # keep BEAM_SIZE best trees at each CYK step


def main():
    # load dataset
    if DEBUG:
        print('Loading dataset...', end='', flush=True)
        t0 = time.time()

    with open('sequoia-corpus+fct.mrg_strict.txt', 'r') as f:
        data = f.readlines()

    data = list(map(remove_functional_labels, data))

    n_data = len(data)
    n_train = int(n_data * 0.8) if not DEBUG else 10
    n_dev = int(n_data * 0.1)
    n_test = int(n_data * 0.1)
    train_data = data[:n_train]
    dev_data = data[n_train:n_train+n_dev]
    test_data = data[n_train+n_dev:]

    if DEBUG:
        print(f'done in {time.time() - t0}s')
        print(f'-> dataset of size {n_data} ({n_train} train, {n_dev} dev, {n_test} test)')

    input_file = 'evaluation_data.parser_input'
    target_file = 'evaluation_data.parser_target'
    if not os.path.isfile(input_file) or not os.path.isfile(target_file):
        print(f'Creating files "{input_file}" and "{target_file}"...', end='', flush=True)
        with open(input_file, 'w') as f:
            for sentence in list(map(extract_sentence, test_data)):
                f.write(sentence + '\n')
        with open(target_file, 'w') as f:
            for sentence in test_data:
                f.write(sentence)
        print('done')
        print('Stopping, please run the code again')
        return

    # generate grammar in chomsky normal form
    if DEBUG:
        print('Generating grammar...', end='', flush=True)
        t0 = time.time()

    productions = []
    for s in train_data:
        s_tree = Tree.fromstring(s, remove_empty_top_bracketing=True)
        s_tree.chomsky_normal_form()
        s_tree.collapse_unary(collapsePOS=True, collapseRoot=True, joinChar='@')
        productions += s_tree.productions()
    start_state = Nonterminal('SENT')
    grammar = induce_pcfg(start_state, productions)

    assert grammar.is_chomsky_normal_form()

    # compute sets of all start symbols, non terminals symbols and terminal words
    start_symbols = set()  # all symbols SENT and SENT@... produced by tree.collapse_unary
    symbols = set()
    lexicon = set()

    for prod in grammar.productions():
        symbols.add(prod.lhs())
        for tk in prod.rhs():
            if type(tk) is str: lexicon.add(tk)
            elif type(tk) is Nonterminal: symbols.add(tk)
        if str(prod.lhs()).startswith(str(start_state) + '@'):
            start_symbols.add(prod.lhs())
    start_symbols.add(start_state)

    # create a unique id for each symbol
    tk2id = {}
    id2tk = {}
    id = 0
    for tk in symbols:
        tk2id[tk] = id
        id2tk[id] = tk
        id += 1
    
    # make list of start ids to look for in final CYK array
    start_ids = []
    for start_symbol in start_symbols:
        start_ids.append(tk2id[start_symbol])
    start_ids = np.array(start_ids)

    if DEBUG:
        print(f'done in {time.time() - t0}s')
        print(f'-> {len(symbols)} non terminal symbols, {len(lexicon)} words')

    # load sentences to parse
    if DEBUG:
        print('Loading sentences to parse...', end='',)
        t0 = time.time()

    # sentences = list(map(extract_sentence, train_data))
    # targets = [s.strip() for s in train_data]

    sentences = []

    try:
        s = input()
        while s:
            sentences.append(s)
            s = input()
    except EOFError:
        # no more input to read (happens when reading from file)
        pass

    assert len(sentences) > 0
    sentences = [s.split() for s in sentences]

    if DEBUG:
        print(f'done in {time.time() - t0}s')
        print(f'-> {len(sentences)} sentences')

    # initialize CYK parser
    if DEBUG:
        print('Initializing CYK parser...', end='', flush=True)
        t0 = time.time()

    cyk_probas = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for prod in grammar.productions():
        if len(prod) == 2:
            cyk_probas[prod.lhs()][prod.rhs()[0]][prod.rhs()[1]] = prod.prob()

    n_symbols = len(symbols)
    n_max = max(list(map(len, sentences)))
    cyk_dp_probas = np.zeros((n_max, n_max, n_symbols), dtype=np.float)
    cyk_dp_partition = np.zeros((n_max, n_max, n_symbols, 3), dtype=np.int)

    if DEBUG:
        print(f'done in {time.time() - t0}s')
    
    # initialize OOV module
    if DEBUG:
        print('Initializing OOV module...', end='', flush=True)
        t0 = time.time()

    oov_lexicon, oov_embeddings = pickle.load(open('polyglot-fr.pkl', 'rb'), encoding='latin1')
    word2embedding = {}  # word -> embedding for words in OOV lexicon
    oov_corpus_embeddings = {}  # word idx -> embedding for words in the intersection of OOV and corpus lexicons
    oov_corpus_words = {}  # word idx -> word for words in the intersection of OOV and corpus lexicons
    n_oov_corpus = 0
    for i, word in enumerate(oov_lexicon):
        if word in lexicon:
            oov_corpus_embeddings[n_oov_corpus] = oov_embeddings[i]
            oov_corpus_words[n_oov_corpus] = word
            n_oov_corpus += 1
        word2embedding[word] = oov_embeddings[i]
    build_tree = np.zeros((n_oov_corpus, oov_embeddings.shape[1]))
    for i in range(n_oov_corpus):
        build_tree[i,:] = oov_corpus_embeddings[i]
    oov_tree = KDTree(build_tree)

    new_words = {}  # unknown word -> new word to avoid searching for new word several times for same unknown word

    if DEBUG:
        print(f'done in {time.time() - t0}s')
        print(f'-> OOV lexicon of {len(oov_lexicon)} words embedded over {oov_embeddings.shape[1]} features')
        
    # parse sentences (and un-chomsky them)
    for sentence in sentences:
        sentence0 = list(sentence)
        assert type(sentence) is list
        n = len(sentence)
        
        # deal with OOV words
        for i, word in enumerate(sentence):
            if word not in lexicon:
                # unknown word
                if DEBUG:
                    print(f'"{word}" is not in lexicon', end=' ')
                if word in new_words:
                    if DEBUG:
                        print(f'but we already changed it to {new_words[word]} before')
                    sentence[i] = new_words[word]
                elif word in oov_lexicon:
                    if DEBUG:
                        print('but is in oov lexicon')
                    # word is in OOV lexicon -> we exchange it with the closest
                    # word in corpus lexicon (in terms of cosine similarity on embeddings)
                    best_j = -1
                    best_dist = 1e9
                    for j, embedding in oov_corpus_embeddings.items():
                        if word != oov_corpus_words[j]:
                            dist = cosine(embedding, word2embedding[word])
                            if dist < best_dist:
                                best_dist = dist
                                best_j = j
                    new_word = oov_corpus_words[best_j]
                    new_words[word] = new_word
                    sentence[i] = new_word
                    if DEBUG:
                        print(f'word has been replaced with "{new_word}" by closest embedding search')
                else:
                    if DEBUG:
                        print('nor in oov lexicon')
                    # word is in neither lexicon -> try to correct its spelling by taking minimum
                    # levenshtein dist wrt oov lexicon, then look for closest embedding in corpus lexicon

                    # correct spelling to obtain a word in oov lexicon
                    best_word = ''
                    best_dist = 99999
                    for word_lex in oov_lexicon:
                        dist = fast_levenshtein_dist(word, word_lex)  # levenshtein_dist(word, word_lex)
                        if dist < best_dist:
                            best_dist = dist
                            best_word = word_lex

                    # get nearest word in corpus lexicon
                    if best_word in lexicon:
                        new_word = best_word
                    else:
                        best_j = -1
                        best_dist = 1e9
                        for j, embedding in oov_corpus_embeddings.items():
                            if best_word != oov_corpus_words[j]:
                                dist = cosine(embedding, word2embedding[best_word])
                                if dist < best_dist:
                                    best_dist = dist
                                    best_j = j
                        new_word = oov_corpus_words[best_j]
                    new_words[word] = new_word
                    new_words[best_word] = new_word
                    sentence[i] = new_word
                    if DEBUG:
                        print(f'word has been replaced with "{new_word}" by closest embedding search on spelling correction "{best_word}"')

        # reset dp arrays
        cyk_dp_probas.fill(0)
        cyk_dp_partition.fill(0)

        # initialize dictionary for beam search
        probas_dict = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                probas_dict[i,j] = {}
        beam_size = BEAM_SIZE

        # cyk init
        if DEBUG:
            print(1, '/', n)
        for i, word in enumerate(sentence):
            for prod in grammar.productions(rhs=word):
                cyk_dp_probas[0,i,tk2id[prod.lhs()]] = prod.prob()*10
            probas_dict[0,i] = {id2tk[idx]: cyk_dp_probas[0,i,idx] for idx in np.argsort(cyk_dp_probas[0,i,:])[-beam_size:]}

        # cyk DP loop
        for l in range(1, n):  # length
            if DEBUG:
                print(l+1, '/', n)
            for s in range(n - l):  # start
                for p in range(l):  # partition
                    # for lhs in cyk_probas:
                    #     for rhs1 in cyk_probas[lhs]:
                    #         for rhs2 in cyk_probas[lhs][rhs1]:
                    for rhs1 in probas_dict[p,s]:
                        for prod in grammar.productions(rhs=rhs1):
                            if len(prod) == 2:
                                lhs = prod.lhs()
                                rhs2 = prod.rhs()[1]
                                if rhs2 in probas_dict[l-p-1,s+p+1]:
                                # for rhs2 in probas_dict[l-p-1,s+p+1]:
                                    # prob_prod = cyk_probas[lhs][rhs1][rhs2]
                                    prob_prod = prod.prob()
                                    prob_rhs1 = cyk_dp_probas[p,s,tk2id[rhs1]]
                                    prob_rhs2 = cyk_dp_probas[l-p-1,s+p+1,tk2id[rhs2]]
                                    prob = prob_rhs1 * prob_rhs2 * prob_prod * 1000
                                    if prob > cyk_dp_probas[l,s,tk2id[lhs]]:
                                        cyk_dp_probas[l,s,tk2id[lhs]] = prob
                                        cyk_dp_partition[l,s,tk2id[lhs],0] = p
                                        cyk_dp_partition[l,s,tk2id[lhs],1] = tk2id[rhs1]
                                        cyk_dp_partition[l,s,tk2id[lhs],2] = tk2id[rhs2]
                probas_dict[l,s] = {id2tk[idx]: cyk_dp_probas[l,s,idx] for idx in np.argsort(cyk_dp_probas[l,s,:])[-beam_size:]}

        # extract best tree
        id_best_tree = start_ids[np.argmax(cyk_dp_probas[n-1,0,start_ids])]
        if cyk_dp_probas[n-1,0,id_best_tree] > 1e-10:
            parsed_str = extract_parse_tree(sentence0, cyk_dp_partition, 0, n - 1, id_best_tree, id2tk, init=True)
            parsed_str = beautify(parsed_str)
            parsed_tree = Tree.fromstring(parsed_str, remove_empty_top_bracketing=False)
            parsed_tree.un_chomsky_normal_form(unaryChar='@')
            s_output = parsed_tree.pformat(margin=int(1e10)).strip()
            # print result on stdout
            print(s_output)
        else:
            # could not parse sentence: print blank line
            print('-') 


if __name__ == '__main__':
    main()