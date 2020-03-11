from grammar import PCFG, Token, Production
import numpy as np
from collections import defaultdict


class Parser(object):
    """Probabilistic CYK parser"""

    def __init__(self, grammar, beam_size=5):
        self.grammar = grammar
        self.beam_size = beam_size

        self.probas = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for lhs in self.grammar.all_symbols:
            for rhs in self.grammar.all_symbols:
                for prod in self.grammar.productions(lhs=lhs, rhs=rhs):
                    if len(prod.rhs) == 2:
                        rhs1, rhs2 = prod.rhs
                        self.probas[lhs][rhs1][rhs2] = prod.prob
        
        self.tk2id = lambda symbol: self.grammar.symbol2id[symbol]
        self.id2tk = lambda id: self.grammar.id2symbol[id]


    def parse(self, sentence):
        if type(sentence) is str:
            sentence = sentence.split()

        probas, partition_max = self._CYK(sentence)
        start_id = self.tk2id(self.grammar.start_symbol)
        # print(probas[-1,0,start_id])
        if probas[-1,0,start_id] > 1e-10:
            parsed_sentence = self._extract_parse_tree(sentence, partition_max, 0, len(sentence) - 1, start_id)
            parsed_sentence = [self.grammar.start_symbol, parsed_sentence]
            parsed_sentence = self._remove_artificial_symbols(parsed_sentence)
            parsed_sentence = self._beautify(parsed_sentence)
            return parsed_sentence
        return None

    def _remove_artificial_symbols(self, parsed):
        assert type(parsed) is list
        out = []

        for i in range(len(parsed)):
            if type(parsed[i]) is list:
                parsed_i = self._remove_artificial_symbols(parsed[i])
                if len(parsed_i) > 1:
                    out.append(parsed_i)
                else:
                    out += parsed_i
            elif parsed[i] not in self.grammar.artificial_symbols:
                out.append(parsed[i])

        if len(out) == 1:
            return out[0]
        return out
    
    def _beautify(self, sentence):
        sentence = str(sentence)
        sentence = sentence.replace('[', ' ( ')
        sentence = sentence.replace('],', ' ) ')
        sentence = sentence.replace(']', ' ) ')
        sentence = sentence.strip().split()
        for i, w in enumerate(sentence):
            if w[-1] == ',':
                sentence[i] = w[:-1]
            elif w[0] == '\'' and w[-1] == '\'':
                sentence[i] = w[1:-1]
        for i in range(1, len(sentence)):
            if sentence[i-1] == '(' and sentence[i] == '(':
                sentence[i-1] = ''
                depth = 1
                for j in range(i, len(sentence)):
                    if sentence[j] == '(':
                        depth += 1
                    elif sentence[j] == ')':
                        depth -= 1
                    if depth == 0:
                        sentence[j] = ''
                        break
        sentence = [w for w in sentence if w]
        sentence = '(' + ' '.join(sentence).strip() + ')'
        for c in [' ) ', ') ', ' )']:
            sentence = sentence.replace(c, ')')
        for c in [' ( ', '( ', ' (']:
            sentence = sentence.replace(c, '(')
        sentence = sentence.replace('(', ' (').strip()

        return sentence

    def _extract_parse_tree(self, sentence, partition_max, start, length, id_symbol):
        if length == 0:
            return Token(sentence[start], terminal=True)

        p, rhs1_id, rhs2_id = partition_max[length,start,id_symbol,:]
        rhs1 = self.id2tk(rhs1_id)
        rhs2 = self.id2tk(rhs2_id)

        parse1 = self._extract_parse_tree(sentence, partition_max, start, p, rhs1_id)
        parse2 = self._extract_parse_tree(sentence, partition_max, start + p + 1, length - p - 1, rhs2_id)

        return [[rhs1, parse1], [rhs2, parse2]]

    def _CYK(self, sentence):
        # print(sentence)
        n = len(sentence)

        # init arrays
        probas = np.zeros((n, n, self.grammar.n_all_symbols), dtype=np.float)
        partition_max = np.zeros((n, n, self.grammar.n_all_symbols, 3), dtype=np.int)

        for i, word in enumerate(sentence):
            for prod in self.grammar.productions(rhs=Token(word, terminal=True)):
                probas[0,i,self.tk2id(prod.lhs)] = prod.prob*100

        # probas[length, start, symbol]
        # for l in range(1, n):  # length
        #     print(l+1, '/', n)
        #     for s in range(n - l):  # start
        #         for p in range(l):  # partition
        #             for lhs in self.probas:
        #                 for rhs1 in self.probas[lhs]:
        #                     for rhs2 in self.probas[lhs][rhs1]:
        #                         if self.probas[lhs][rhs1][rhs2] > 1e-10:
        #                             prob_prod = self.probas[lhs][rhs1][rhs2]
        #                             prob_rhs1 = probas[p,s,self.tk2id(rhs1)]
        #                             prob_rhs2 = probas[l-p-1,s+p+1,self.tk2id(rhs2)]
        #                             prob = prob_rhs1 * prob_rhs2 * prob_prod *1000000
        #                             if prob > probas[l,s,self.tk2id(lhs)]:
        #                                 probas[l,s,self.tk2id(lhs)] = prob
        #                                 partition_max[l,s,self.tk2id(lhs),0] = p
        #                                 partition_max[l,s,self.tk2id(lhs),1] = self.tk2id(rhs1)
        #                                 partition_max[l,s,self.tk2id(lhs),2] = self.tk2id(rhs2)
                    
        # probas[length, start, symbol]

        probas_dict = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                probas_dict[i,j] = {}

        for i, word in enumerate(sentence): #init copied tmp
            for prod in self.grammar.productions(rhs=Token(word, terminal=True)):
                probas[0,i,self.tk2id(prod.lhs)] = prod.prob*10
                # probas_dict[0,i][prod.lhs] = prod.prob*100

            probas_dict[0,i] = {self.id2tk(idx): probas[0,i,idx] for idx in np.argsort(probas[0,i,:])[-self.beam_size:]}

        for l in range(1, n):  # length
            # print(l+1, '/', n)
            for s in range(n - l):  # start
                for p in range(l):  # partition
                    # handle binary rules
                    for rhs1 in probas_dict[p,s]:
                        for prod in self.grammar.productions(rhs=rhs1):
                            if len(prod.rhs) == 2:
                            #for lhs in self.grammar.all_symbols:
                                lhs = prod.lhs
                                for rhs2 in probas_dict[l-p-1,s+p+1]:
                                    # print(lhs, rhs1, rhs2)
                                    prob_prod = self.probas[lhs][rhs1][rhs2]
                                    prob_rhs1 = probas[p,s,self.tk2id(rhs1)]
                                    prob_rhs2 = probas[l-p-1,s+p+1,self.tk2id(rhs2)]
                                    prob = prob_rhs1 * prob_rhs2 * prob_prod * 1000
                                    if prob > probas[l,s,self.tk2id(lhs)]:
                                        probas[l,s,self.tk2id(lhs)] = prob
                                        partition_max[l,s,self.tk2id(lhs),0] = p
                                        partition_max[l,s,self.tk2id(lhs),1] = self.tk2id(rhs1)
                                        partition_max[l,s,self.tk2id(lhs),2] = self.tk2id(rhs2)
                                        # probas_dict[l,s][lhs] = prob
                    
                    # try to handle unit rules
                    # for rhs in probas_dict[l,s]:
                    #     # print('hi')
                    #     for prod in self.grammar.productions(rhs=rhs):
                    #         if len(prod.rhs) == 1:
                    #             lhs = prod.lhs
                    #             prob_rhs = probas[l,s,self.tk2id(rhs)]
                    #             prob = prob_rhs * prod.prob * 1000000
                    #             if prob > probas[l,s,self.tk2id(lhs)]:
                    #                 probas[l,s,self.tk2id(lhs)] = prob
                    #                 partition_max[l,s,self.tk2id(lhs),0] = -1
                    #                 partition_max[l,s,self.tk2id(lhs),1] = self.tk2id(rhs)
                    #                 partition_max[l,s,self.tk2id(lhs),2] = -1

                probas_dict[l,s] = {self.id2tk(idx): probas[l,s,idx] for idx in np.argsort(probas[l,s,:])[-self.beam_size:]}
                    
                    # probas[l,s,np.argsort(probas[l,s,:])[:-self.beam_size]] = 0

        # print(np.mean(probas[-1,0,:]), self.id2tk(np.argmax(probas[-1,0,:])), probas[-1,0,self.tk2id(self.grammar.start_symbol)])
        return probas, partition_max