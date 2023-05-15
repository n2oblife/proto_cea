import sys
sys.path.append('/home/zk274707/Projet/proto/env_proto/lib/python3.10/site-packages/trankit-1.1.0-py3.10.egg')

from trankit import Pipeline

# initialize a pipeline for English
p = Pipeline('english', gpu=False)

# a non-empty string to process, which can be a document or a paragraph with multiple sentences
doc_text = "Hello! This is Trankit. Hello I'm a single sentence. And another sentence. And the very very last one"

print(f"First input is doc_text = {doc_text} of length {len(doc_text)} which elements are str, here is 5 sentences for 1 input in a single str",file=sys.stderr)
all = p(doc_text)
print(all)
print("-----------------------------DONE-----------------------------")
# print(p._config.__dict__)