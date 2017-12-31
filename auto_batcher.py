class AutoBatcher:
    """
    A class for naive auto batching variable length sequences by length
    minibatch guarentees that the batch returned will not exceed the specified length.
    The longer the sequence the less value this provides but it is good for processing sentences
    since senteces tend not to contain more than 30+ words.
    """
    def __init__(self, X, y, batch_size, shuffle=False):
        self.data = zip(X, y)
        self.X = X
        self.y = y
        self.shuffle = shuffle
        self.batch_size = batch_size
        self._batches = []
        self._batch()

    def batch_count(self):
        """
        returns num of batches
        """
        return len(self._batches)

    def get_batches(self):
        """
        Return loaders
        """
        if self.shuffle:
            random.shuffle(self._batches)
        return self._batches

    def _list_to_array(self, list):
        """
        np.array() tries to infer nested dims, this is a simple list to array converter
        """
        arr = np.empty(len(list), dtype=object)
        for i, o in enumerate(list):
            arr[i] = o
        return arr

    def _batch(self):
        """
        Batch the sequences
        """
        if self.batch_size > 1:
            self.data.sort(key=lambda x: len(x[0]))
        groups = [list(group) for key, group in itertools.groupby(self.data, lambda x: len(x[0]))]
        for group in groups:
            if self.shuffle:
                random.shuffle(group)
            for i in xrange(0, len(group), self.batch_size):
                batch = group[i:i + self.batch_size]
                batch_input, batch_labels = zip(*batch)
                self._batches.append([self._list_to_array(list(batch_input)), list(batch_labels)])
