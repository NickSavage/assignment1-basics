import re
import tests.adapters

results, merged_list = tests.adapters.run_train_bpe(
    # "./tests/fixtures/address.txt", 2, []
    # "./tests/fixtures/tinystories_sample_5M.txt",
    "./input",
    256 + 6,
    ["<|endoftext|>"],
)
print(results, merged_list)
