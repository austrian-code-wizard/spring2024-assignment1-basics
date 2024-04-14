"""
- load tokenizer
- load dataset with mmep
- get torch device
- create model
- optionally load model/optimizer from checkpoint
- iterate
- sample batch
- pass to model
- get loss
- optimizer
- if some interval: print loss
- if some interval: save checkpoint
"""
