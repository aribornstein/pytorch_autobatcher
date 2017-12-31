# pytorch autobatcher

An auto batcher for variable sized sequences

A class for auto batching variable length sequences by length minibatch guarantees that the batch returned will not exceed the specified length. The longer the sequence the less value this provides but it is good for processing sentences since sentences tend not to contain more than 30+ words.


## Example Usage
```python

    # Generate example langauge of odd and even sequences
    POS_DATA = [[0 for i in range(random.choice(range(2, 100, 2)))] for _ in range(500)]
    NEG_DATA = [[0 for i in range(random.choice(range(1, 100, 2)))] for _ in range(500)]
    DATA = NEG_DATA + POS_DATA

    # Create gold data for training
    X = [[num for num in seq] for seq in [list(x) for x in DATA]]
    y = [0]*len(NEG_DATA) + [1]*len(POS_DATA)

    # Generate auto batcher
    TRAIN_BATCHER =  AutoBatcher(X, y, batch_size=100, shuffle=True)
    for i, data in enumerate(train_batcher.get_batches(), 0):
        inputs, labels = data
        # Do training here
```
