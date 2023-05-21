def find_word_indices(sentence, word):
    indices = []
    start_index = 0
    while True:
        index = sentence.find(word, start_index)
        if index == -1:
            break
        indices.append((index, index + len(word) - 1))
        start_index = index + len(word) - 1
    return indices
