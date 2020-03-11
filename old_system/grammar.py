from collections import defaultdict


class Token(object):
    def __init__(self, token_str, terminal=False, ignore_functional_labels=True):
        self.token_str = token_str
        self.terminal = terminal

        if not terminal and ignore_functional_labels:
            self.token_str = token_str.partition('-')[0]

    def __str__(self):
        if self.terminal:
            return f'\'{self.token_str}\''
        return self.token_str

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((self.token_str, self.terminal))

    def __eq__(self, tk2):
        return self.token_str == tk2.token_str


class Production(object):
    def __init__(self, lhs, rhs, prob=0):
        self.lhs = lhs
        self.rhs = rhs
        self.prob = prob

        if type(self.rhs) is Token:
            self.rhs = (self.rhs,)
        elif type(self.rhs) is list:
            self.rhs = tuple(self.rhs)

    def __repr__(self):
        rhs_str = ' '.join(list(map(str, self.rhs)))
        return f'{self.lhs} -> {rhs_str} [{self.prob}]'


class PCFG(object):
    def __init__(self, corpus, start_symbol):
        self.corpus = corpus
        self.start_symbol = start_symbol

        self._lhs = defaultdict(set)
        self._rhs = defaultdict(set)

        self._build_grammar()
        self._assert_probabilities()

    def _build_grammar(self):
        # compute (dict: Token -> Token list) representing all productions in corpus
        prod_dict = defaultdict(list)
        for i in range(len(self.corpus)):
            self._extract_productions(i, prod_dict)
        del prod_dict['']

        # compute (dict: Token -> (Token list) list) by merging all rhs for each lhs
        # and dict of (lhs, rhs) counts and dict of lhs counts
        prod_dict_merged = defaultdict(list)
        lhs_rhs_count = defaultdict(int)
        lhs_count = defaultdict(int)
        for lhs, rhs in prod_dict.items():
            prod_dict_merged[lhs[0]].append(rhs)
            lhs_rhs_count[(lhs[0], tuple(rhs))] += 1
            lhs_count[lhs[0]] += 1

        # compute dicts (lhs -> productions) and (first rhs -> productions) for quick access
        for lhs, rhs_list in prod_dict_merged.items():
            for rhs in set(map(tuple, rhs_list)):
                rhs_count = lhs_rhs_count[(lhs, rhs)]
                prod = Production(lhs, rhs, rhs_count)
                self._lhs[lhs].add(prod)
                self._rhs[rhs[0]].add(prod)

        # compute probabilities of all productions
        for lhs, prods in self._lhs.items():
            for prod in prods:
                prod.prob /= lhs_count[lhs]

    def _extract_productions(self, sentence_idx, prod_dict):
        sentence = self.corpus[sentence_idx]

        # make sure parenthesis are surrounded by spaces
        sentence = sentence.replace('(', ' ( ')
        sentence = sentence.replace(')', ' ) ')
        
        lhs_list = ['']  # queue of (left hand sides, unique_id)
        last_is_open = False  # whether last seq is an opening bracket '('

        # parse sentence
        for i, seq in enumerate(sentence.split()):
            unique_id = str(sentence_idx) + '.' + str(i)
            if seq == '(':
                if last_is_open:
                    # if consecutive opening brackets, add blank lhs to be popped later
                    lhs_list.append('')
                last_is_open = True
            elif seq == ')':
                lhs_list.pop()
                last_is_open = False
            else:
                # add seq in the rhs of the last lhs
                # seq is terminal iff the previous seq is not an opening bracket
                prod_dict[lhs_list[-1]].append(Token(seq, terminal=not last_is_open))
                if last_is_open:
                    # if it is not terminal, push seq in the lhs queue
                    lhs_list.append((Token(seq), unique_id))
                last_is_open = False

    def _assert_probabilities(self):
        for lhs in self._lhs.keys():
            total = 0
            for prod in self._lhs[lhs]:
                total += prod.prob
            if abs(total - 1) >= 1e-7:
                print("doesn't add up:", lhs, total)
            assert(abs(total - 1) < 1e-7)

    def chomsky_normal_form(self):
        # follows the steps explained on Wikipedia
        # https://en.wikipedia.org/wiki/Chomsky_normal_form#Converting_a_grammar_to_Chomsky_normal_form

        new_char = '$'  # make sure this symbol is not already in the tokens

        # step 1: START

        # old_start = self.start_symbol
        # self.start_symbol = Token(new_char + old_start.token_str)

        # new_prod = Production(self.start_symbol, old_start, 1)
        # self._lhs[new_prod.lhs] = set([new_prod])
        # self._rhs[new_prod.rhs[0]] = set([new_prod])

        # step 2: TERM

        new_prods = defaultdict(lambda: defaultdict(int))
        for _, prods in self._lhs.items():
            for prod in prods:
                if len(prod.rhs) > 1:
                    for rhs in prod.rhs:
                        if rhs.terminal:
                            terminal_str = rhs.token_str
                            rhs.terminal = False
                            rhs.token_str = new_char + terminal_str
                            new_prods[rhs.token_str][terminal_str] += 1

        for lhs_str, rhs_lst in new_prods.items():
            total = sum(rhs_lst.values())
            lhs = Token(lhs_str)
            for rhs_str, qty in rhs_lst.items():
                rhs = Token(rhs_str, terminal=True)
                new_prod = Production(lhs, rhs, prob=qty/total)
                self._lhs[lhs].add(new_prod)
                self._rhs[rhs].add(new_prod)        

        # step 3: BIN

        new_prods = defaultdict(lambda: defaultdict(int))

        prods_to_add = []
        prods_to_remove = []

        for lhs, prods in self._lhs.items():
            for prod in prods:
                if len(prod.rhs) > 2:
                    tk_strs = [rhs.token_str for rhs in prod.rhs]
                    # add first rule A -> X1 A1
                    A = new_char.join([lhs.token_str] + tk_strs[1:])
                    new_prod = Production(lhs, (Token(tk_strs[0]), Token(A)), prob=prod.prob)
                    # self._lhs[lhs].add(new_prod)
                    # self._rhs[Token(tk_strs[0])].add(new_prod)
                    prods_to_add.append(new_prod)
                    # remove old rule A -> X1 X2 ... Xn
                    # self._lhs[lhs].remove(prod)
                    # self._rhs[prod.rhs[0]].remove(prod)
                    prods_to_remove.append(prod)
                    # add other rules Ai -> Xi+1 Ai+1
                    for i in range(1, len(prod.rhs) - 2):
                        new_A = new_char.join([lhs.token_str] + tk_strs[i+1:])
                        new_prods[A][(tk_strs[i], new_A)] += 1
                        A = new_A
                    # add last rule An-2 -> Xn-1 Xn
                    new_prods[A][(tk_strs[-2], tk_strs[-1])] += 1

        for prod in prods_to_remove:
            self._lhs[prod.lhs].discard(prod)
            self._rhs[prod.rhs[0]].discard(prod)
            if len(self._lhs[prod.lhs]) == 0:
                del self._lhs[prod.lhs]
            if len(self._rhs[prod.rhs[0]]) == 0:
                del self._rhs[prod.rhs[0]]

        for prod in prods_to_add:
            self._lhs[prod.lhs].add(prod)
            self._rhs[prod.rhs[0]].add(prod)

        for lhs_str, rhs_dict in new_prods.items():
            total = sum(rhs_dict.values())
            lhs = Token(lhs_str)
            for rhs_str, qty in rhs_dict.items():
                rhs_lst = list(map(Token, rhs_str))
                new_prod = Production(lhs, rhs_lst, prob=qty/total)
                self._lhs[lhs].add(new_prod)
                self._rhs[rhs_lst[0]].add(new_prod)

        # step 4: DEL
        if len(self._rhs[Token('')]) > 0 or len(self._rhs[Token('', terminal=True)]) > 0:
            raise ValueError('Non implemented')

        # step 5: UNIT

        if False:
            rhs_contains = defaultdict(set)
            for lhs, prods in self._lhs.items():
                for prod in prods:
                    for rhs in prod.rhs:
                        rhs_contains[rhs].add(prod)

        done = False
        while not done:
            new_prods = defaultdict(lambda: defaultdict(float))
            prods_to_remove = []
            done = True
            for lhs, prods in self._lhs.items():
                for prod in prods:
                    if len(prod.rhs) == 1:
                        rhs = prod.rhs[0]
                        if not rhs.terminal:
                            if rhs in self._lhs:
                                if True:
                                    for prod2 in self._lhs[rhs]:
                                        prob = prod.prob * prod2.prob
                                        new_prods[lhs][prod2.rhs] += prob
                                        done = False
                                    prods_to_remove.append(prod)

                                if False:
                                    new_tk = Token(lhs.token_str + '+' + rhs.token_str)
                                    if lhs == self.start_symbol:
                                        print('initial production:', prod)
                                    # 1)
                                    prob_sum = 0
                                    for prod2 in self._lhs[rhs]:
                                        if lhs == self.start_symbol:
                                            prob = prod.prob * prod2.prob
                                            new_prods[lhs][prod2.rhs] += prob
                                            print('new prod:', lhs, prod2.rhs, prob)
                                            prob_sum += prob
                                        else:
                                            prob = prod2.prob
                                            new_prods[new_tk][prod2.rhs] += prob
                                            # print('new prod:', new_tk, prod2.rhs, prob)
                                        # done = False
                                    if lhs == self.start_symbol:
                                        print('total:', prob_sum)
                                    if lhs != self.start_symbol:
                                        # 2)
                                        #for _, prods2 in self._lhs.items():
                                        for prod2 in rhs_contains[lhs]:
                                            if lhs in prod2.rhs:
                                                prod2.prob = prod2.prob * (1 - prod.prob)
                                                if len(prod2.rhs) == 1:
                                                    new_prods[lhs][(new_tk,)] += prod2.prob * prod.prob
                                                elif len(prod2.rhs) == 2:
                                                    if prod2.rhs[0] == lhs and prod2.rhs[1] == lhs:
                                                        new_prods[lhs][(lhs,new_tk)] += prod2.prob * prod.prob * (1 - prod.prob) / 2
                                                        new_prods[lhs][(new_tk,lhs)] += prod2.prob * prod.prob * (1 - prod.prob) / 2
                                                        new_prods[lhs][(new_tk,new_tk)] += prod2.prob * prod.prob * prod.prob
                                                    elif prod2.rhs[0] == lhs:
                                                        new_prods[lhs][(new_tk,prod2.rhs[1])] += prod2.prob * prod.prob
                                                    elif prod2.rhs[1] == lhs:
                                                        new_prods[lhs][(prod2.rhs[0],new_tk)] += prod2.prob * prod.prob                                                    
                                                else:
                                                    raise ValueError('Grammar should only have unit and binary rules')
                                        # 3)
                                        for prod2 in self._lhs[lhs]:
                                            if prod2.rhs != prod.rhs:
                                                prod2.prob /= (1 - prod.prob)
                                    # 4)
                                    prods_to_remove.append(prod)

            """
            For each A -> B with B non terminal, do:

            1) For each B -> sth, I add a rule A+B -> sth with prob(A+B -> sth) = prob(B -> sth)
            2) For each rule X -> X1...A..Xn, create a new rule X -> X1..A+B..Xn
                and set prob(X -> X1...A..Xn) = prob(X -> X1...A..Xn) * (1 - prob(A -> B)) and prob(X -> X1...A+B...Xn) = prob(X -> X1...A..Xn) * prob(A -> B)
            3) For all rules A -> sth, set prob(A -> sth) = prob(A -> sth) / (1 - prob(A -> B))
            4) Remove A -> B

            Note: if A is SENT, do not rename it into A+B (thus no need for 2 and 3)
                    and prob(SENT -> sth) = prob(SENT -> B) * prob(B -> sth)
            """

            for prod in prods_to_remove:
                # if prod.lhs == self.start_symbol:
                #     print('REMOVE', prod)
                self._lhs[prod.lhs].discard(prod)
                self._rhs[prod.rhs[0]].discard(prod)
                if len(self._lhs[prod.lhs]) == 0:
                    del self._lhs[prod.lhs]
                if len(self._rhs[prod.rhs[0]]) == 0:
                    del self._rhs[prod.rhs[0]]

            for lhs, rhs_dict in new_prods.items():
                for rhs, prob in rhs_dict.items():
                    new_prod = Production(lhs, rhs, prob=prob)
                    # if lhs == self.start_symbol:
                    #     print('ADD', new_prod)
                    self._lhs[lhs].add(new_prod)
                    self._rhs[rhs[0]].add(new_prod)
        
        # last step: verifications
        self._assert_probabilities()
        self._assert_chomsky_normal_form()

    def _assert_chomsky_normal_form(self):
        for _, prods in self._lhs.items():
            for prod in prods:
                assert not prod.lhs.terminal
                rhs = prod.rhs
                assert len(rhs) == 1 or len(rhs) == 2
                if len(rhs) == 1:
                    assert rhs[0].terminal
                else:
                    assert not (rhs[0].terminal or rhs[1].terminal)

    def compute_symbols(self):
        """Get all symbols and artificial symbols in grammar"""
        self.all_symbols = set()
        self.artificial_symbols = set()  # symbols introduced by chomsky normal form
        self.corpus_symbols = set()  # symbols present in the original corpus
        self.terminals = set()  # vocabulary

        # all symbols
        for _, prods in self._lhs.items():
            for prod in prods:
                self.all_symbols.add(prod.lhs)
                for rhs in prod.rhs:
                    if rhs.terminal:
                        self.terminals.add(rhs)
                    else:
                        self.all_symbols.add(rhs)

        # corpus symbols
        whole_corpus = '\n'.join(self.corpus)
        for symbol in self.all_symbols:
            if not symbol.terminal and symbol.token_str in whole_corpus:
                self.corpus_symbols.add(symbol)
        
        # artificial symbols
        self.artificial_symbols = self.all_symbols - self.corpus_symbols
        
        # count symbols
        self.n_all_symbols = len(self.all_symbols)
        self.n_artificial_symbols = len(self.artificial_symbols)
        self.n_corpus_symbols = len(self.corpus_symbols)
        self.n_terminals = len(self.terminals)

        # define symbol <-> id dicts
        self.symbol2id = {}
        self.id2symbol = {}
        i = 0
        for symbol in self.all_symbols:
            self.symbol2id[symbol] = i
            self.id2symbol[i] = symbol
            i += 1

    def compute_productions(self):
        self.all_productions = set()
        for _, prods in self._lhs.items():
            for prod in prods:
                self.all_productions.add(prod)
        self.n_productions = len(self.all_productions)

    def productions(self, lhs=None, rhs=None):
        """Return grammar productions, filtered by lhs and/or first rhs"""
        if lhs is None and rhs is None:
            return self.all_productions
        if lhs is None:
            return self._rhs[rhs]
        if rhs is None:
            return self._lhs[lhs]
        return self._lhs[lhs] & self._rhs[rhs]
