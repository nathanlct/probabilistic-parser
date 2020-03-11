import numpy as np


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


def remove_functional_labels(sentence):
    """remove functional labels from SEQUOIA sentence, ie transform NP-SUJ into NP"""
    out = ''
    remove = 0
    for c in sentence:
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


def extract_parse_tree(sentence, partition_max, start, length, id_symbol, id2tk, init=False):
    if length == 0:
        if init:
            return (id2tk[id_symbol], sentence[start])
        return sentence[start]

    p, rhs1_id, rhs2_id = partition_max[length,start,id_symbol,:]
    rhs1 = id2tk[rhs1_id]
    rhs2 = id2tk[rhs2_id]

    parse1 = extract_parse_tree(sentence, partition_max, start, p, rhs1_id, id2tk)
    parse2 = extract_parse_tree(sentence, partition_max, start + p + 1, length - p - 1, rhs2_id, id2tk)
    
    if init:
        return (id2tk[id_symbol], (rhs1, parse1), (rhs2, parse2))
    return ((rhs1, parse1), (rhs2, parse2))


def beautify(sentence):
    # beautify parsed sentence (from recursive tuples to SEQUOIA format)
    parsed_str = str(sentence)
    # split sentence and remove commas
    parsed_str = parsed_str.replace('(', ' ( ')
    parsed_str = parsed_str.replace('),', ' ) ')
    parsed_str = parsed_str.replace(')', ' ) ')
    parsed_str = parsed_str.split()
    # remove quotes around words
    for i, w in enumerate(parsed_str):
        if w[-1] == ',':
            parsed_str[i] = w[:-1]
        elif (w[0] == '\'' and w[-1] == '\'') or (w[0] == '"' and w[-1] == '"'):
            parsed_str[i] = w[1:-1]
    # remove useless parenthesis
    for i in range(1, len(parsed_str)):
        if parsed_str[i-1] == '(' and parsed_str[i] == '(':
            parsed_str[i-1] = ''
            depth = 1
            for j in range(i, len(parsed_str)):
                if parsed_str[j] == '(':
                    depth += 1
                elif parsed_str[j] == ')':
                    depth -= 1
                if depth == 0:
                    parsed_str[j] = ''
                    break
    # get new sentence 
    parsed_str = [w for w in parsed_str if w]
    parsed_str = '(' + ' '.join(parsed_str).strip() + ')'
    # remove most spaces
    for c in [' ) ', ') ', ' )']:
        parsed_str = parsed_str.replace(c, ')')
    for c in [' ( ', '( ', ' (']:
        parsed_str = parsed_str.replace(c, '(')
    # add minimal required spaces
    parsed_str = parsed_str.replace('(', ' (').strip()
    return parsed_str


def levenshtein_dist(s1, s2):
    """
    return minimal number of operations on s1 to transform it into s2
    available operations  (all of cost 1) are:
    (1) deletion of 1 character
    (2) insertion of 1 character
    (3) substitution of 1 character
    """
    n1, n2 = len(s1), len(s2)
    
    # dp[i,j] = levenshtein distance between s1[:i] and s2[:j]
    dp = np.zeros((n1+1, n2+1))
    
    dp[:,0] = np.arange(n1+1)
    dp[0,:] = np.arange(n2+1)
    
    for i in range(1, n1+1):
        for j in range(1, n2+1):
            dp[i,j] = min(dp[i-1,j] + 1, # (1)
                          dp[i,j-1] + 1,  # (2)
                          dp[i-1,j-1] + int(s1[i-1] != s2[j-1]))  # (3)

    return dp[-1,-1]


def fast_levenshtein_dist(source, target):
    """
    CODE TAKEN FROM
    https://github.com/obulkin/string-dist/blob/master/stringdist/pystringdist/rdlevenshtein.py
    (because my levenshtein_dist was too slow)
    """
    s_range = range(len(source) + 1)
    t_range = range(len(target) + 1)
    matrix = [[(i if j == 0 else j) for j in t_range] for i in s_range]
    for i in s_range[1:]:
        for j in t_range[1:]:
            del_dist = matrix[i - 1][j] + 1
            ins_dist = matrix[i][j - 1] + 1
            sub_trans_cost = 0 if source[i - 1] == target[j - 1] else 1
            sub_dist = matrix[i - 1][j - 1] + sub_trans_cost
            matrix[i][j] = min(del_dist, ins_dist, sub_dist)
    return matrix[len(source)][len(target)]