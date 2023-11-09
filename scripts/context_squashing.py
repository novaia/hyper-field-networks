import numpy as np

sequence_length = 524_440
context_length = 120_000

print('Sequence length', sequence_length)
print('Context length', context_length)

naive_padding_size = (context_length - (sequence_length % context_length)) % context_length
naive_padded_sequence_length = sequence_length + naive_padding_size
num_contexts = naive_padded_sequence_length // context_length
context_overlap = naive_padding_size // (num_contexts - 1)

print('Naive padding size:', naive_padding_size)
print('Num contexts', num_contexts)
print('Context overlap', context_overlap)

context_indices = np.arange(num_contexts)
context_start_positions = context_indices * (context_length - context_overlap)
context_end_positions = context_start_positions + context_length
padded_sequence_length = context_end_positions[-1]
squashed_padding_size = padded_sequence_length - sequence_length

print('Context indices', context_indices)
print('Context start positions', context_start_positions)
print('Context end positions', context_end_positions)
print('Padded sequence length', padded_sequence_length)
print('Squashed padding size', squashed_padding_size)