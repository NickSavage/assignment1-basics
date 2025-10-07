import re
import tests.adapters

results, merged_list = tests.adapters.run_train_bpe(
    "./tests/fixtures/tinystories_sample_5M.txt",
    1000,
    ["<|endoftext|>"],
)
for x in merged_list[:30]:
    print(x)

