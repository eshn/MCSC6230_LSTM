import numpy as np

# Takes input directly from fo.readlines()
def parse(text, punc=1):
    text = [line.lower() for line in text]
    delimiters = ['\n', '.', ',', '!', ':', ';', '"', '?', '-', '(', ')', '*' ,'[', ']', '_']
    wordlist = []
    # Puts all lines in one giant list
    lines = []
    for line in text:
        lines.append(line)
    # Extracts words, separated by space, from all lines
    for line in lines:
        words = list(line.split())
        for word in words:
            wordlist.append(word)
        # wordlist.append('\n')
    # Parses the list of words from above by each delimiter
    for delimiter in delimiters:
        old_wordlist = wordlist
        wordlist = []
        for word in old_wordlist:
            if word.find(delimiter) is not -1:
                word2 = list(word.split(delimiter))
                for word in word2:
                    wordlist.append(word)
                if punc == 1:
                    wordlist.append(delimiter)
            else:
                wordlist.append(word)

    # Removes "empty" words
    wordlist = [x for x in wordlist if x != '']
    # Saves the parsed text (order preserving)
    text_parsed = wordlist
    # Appends delimiters to word list
    if punc == 1:
        for delimiter in delimiters:
            wordlist.append(delimiter)
    return sorted(list(set(wordlist))), text_parsed

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)