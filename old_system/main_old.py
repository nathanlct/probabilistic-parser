from grammar import PCFG, Token, Production
from parser import Parser
from PYEVALB import parser as evalb_parser
from PYEVALB import scorer as evalb_scorer

def extract_sentence(sentence):
    """remove labels from SEQUOIA sentence to only keep terminals"""
    sentence = sentence.replace('(', ' ( ')
    sentence = sentence.replace(')', ' ) ')
    out = []
    count = 0
    for seq in sentence.split():
        if seq == '(':
            count = 0
        if count == 2:
            out.append(seq)
        count += 1
    return ' '.join(out)

def remove_functional_labels(s):
    out = ''
    remove = 0
    for c in s:
        if c == ' ':
            remove = 0
        elif c == '(':
            remove = 1
        elif remove == 1 and c == '-':
            remove = 2
        if remove == 2:
            continue
        out += c
    return out



if __name__ == '__main__':
    # retrieve dataset
    print('Loading dataset...', end='', flush=True)

    with open('sequoia-corpus+fct.mrg_strict.txt', 'r') as f:
        data = f.readlines()

    n = len(data)
    n_train = int(n * 0.8)
    n_dev = int(n * 0.1)
    n_test = int(n* 0.1)
    train_data = data[:n_train]
    dev_data = data[n_train:n_train+n_dev]
    test_data = data[n_train+n_dev:]

    # train_data = train_data[:100]

    print('done')
    print(f'-> {n} sentences loaded ({n_train} train, {n_dev} dev, {n_test} test)')

    # create PCFG
    print('Computing PCFG from training corpus...', end='', flush=True)
    grammar = PCFG(train_data, Token('SENT'))        
    print('done')

    # transform PCFG so that it is in Chomsky normal form
    print('Converting PCFG to Chomsky normal form...', end='', flush=True)
    grammar.chomsky_normal_form()
    print('done')

    # compute grammar stats
    print('Retrieving grammar symbols and productions...', end='', flush=True)
    grammar.compute_symbols()
    grammar.compute_productions()
    print('done')
    print(f'-> {grammar.n_all_symbols} non-terminal symbols ({grammar.n_corpus_symbols} originals, '
          f'{grammar.n_artificial_symbols} artificials), {grammar.n_terminals} terminal words')
    print(f'-> {grammar.n_productions} productions')

    # initialize CYK parser
    print('Building parser...', end='', flush=True)
    parser = Parser(grammar, beam_size=10)
    print('done')

    # parse sentences

    total_precision = 0
    total_recall = 0

    for k, sentence in enumerate(train_data):
        # sentence = train_data[4]

        s_input = extract_sentence(sentence)
        s_target = remove_functional_labels(sentence).strip()
        s_output = parser.parse(s_input)

        if not s_output:
            continue

        print('input --> ', s_input)
        print('input labels:', s_target)
        print('output -->', extract_sentence(s_output))
        print('output labels:', s_output)

        target_tree = evalb_parser.create_from_bracket_string(s_target[1:-1])
        output_tree = evalb_parser.create_from_bracket_string(s_output[1:-1])

        # print(target_tree)
        # print(output_tree)

        try:
            s = evalb_scorer.Scorer()
            result = s.score_trees(target_tree, output_tree)

            print(f'sentence {k}, precision={result.prec}, recall={result.recall}')
            total_precision += result.prec
            total_recall += result.recall
            print(f'average so far: precision={total_precision/(k+1)}, recall={total_recall/(k+1)}')
        except:
            print(f'sentence {k}, scorer failed')

        # break