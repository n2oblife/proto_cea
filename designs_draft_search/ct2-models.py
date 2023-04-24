import ctranslate2
import transformers

generator = ctranslate2.Generator("bloom-560m")
tokenizer = transformers.AutoTokenizer.from_pretrained("bigscience/bloom-560m")

text = "Hello, I am"
start_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
results = generator.generate_batch([start_tokens], max_length=30, sampling_topk=10)
print(tokenizer.decode(results[0].sequences_ids[0]))
