puncts = ['?', '!', ':', ';']


def split(text):
    # splits text at period and puncts symbols
    text_spitted = text.split('. ')  # keep expressions like e.g. or q.e.d.
    splitted = []
    for t in text_spitted:
        current = [t]
        search_puncts = puncts[:]
        while len(search_puncts) > 0:
            punct = search_puncts.pop()
            current_splitted = []
            for c in current:
                if punct in c:
                    current_splitted.extend(c.split(punct))
                else:
                    current_splitted.append(c)
            current = current_splitted
        splitted.extend(current)
    splitted = [s.strip() for s in splitted]
    splitted = [s for s in splitted if s != '']
    return [s + '\n' for s in splitted]


# https://stackoverflow.com/a/49186435
def divide(lst, n):  # divide list in n parts of equal length (more or less)
    p = len(lst) // n
    if len(lst) - p > 0:
        return [lst[:p]] + divide(lst[p:], n - 1)
    else:
        return [lst]


class LatexTokenizer:
    """Tokenizer for splitting LaTeX expressions into tokens"""

    def tokenize(self, eq):
        result = []
        last_token = ''
        for char in list(eq):
            if char == '\\':
                if last_token:
                    result.append(last_token)
                last_token = '\\'
            elif last_token and last_token[0] in ['\\']:
                if char.isalpha():
                    last_token += char
                else:
                    if last_token:
                        result.append(last_token)
                    last_token = char
            elif char.isdigit() or char == '.':  # TODO support , as decimal seperator?
                if last_token.replace('.', '').isdigit():
                    last_token += char
                else:
                    if last_token:
                        result.append(last_token)
                    last_token = char
            else:
                if last_token:
                    result.append(last_token)
                last_token = char
        result.append(last_token)
        assert ''.join(result) == eq
        return result

    def num_of_non_whitespace_tokens(self, eq):
        tokenized = self.tokenize(eq)
        cleaned = [t for t in tokenized if t.strip() not in [' ', '{', '}']]
        return len(cleaned)
