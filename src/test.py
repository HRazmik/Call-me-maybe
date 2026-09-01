from llm_sdk import Small_LLM_Model
from vocab import Vocab


model = Small_LLM_Model()
vocab = Vocab(model.get_path_to_vocab_file())
print(f"Loaded vocab with {len(vocab.str_to_id)} tokens.")
print(vocab.str_to_id.items)
