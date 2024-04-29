import numpy as np
from float_tokenization import (
    tokenize, detokenize, bit_count_to_mantissa_shift, bit_count_to_vocab_size
)

mantissa_shift = bit_count_to_mantissa_shift(11)
print(mantissa_shift)

samples_in = np.array([101.93423], dtype=np.float16)
print(samples_in)
tokens_out = tokenize(samples_in, mantissa_shift)
print(tokens_out)
samples = detokenize(tokens_out, mantissa_shift)
print(samples)
print(samples.dtype)

vocab_size = bit_count_to_vocab_size(11)
full_vocab = np.arange(vocab_size-1, dtype=np.uint16)
full_vocab_samples = detokenize(full_vocab, mantissa_shift)
for i in range(full_vocab_samples.shape[0]):
    print(full_vocab[i], full_vocab_samples[i])
