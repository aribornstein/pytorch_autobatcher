# pytorch autobatcher

An auto batcher for variable sized sequences

A class for auto batching variable length sequences by length minibatch guarentees that the batch returned will not exceed the specified length. The longer the sequence the less value this provides but it is good for processing sentences since senteces tend not to contain more than 30+ words.
