from trankit import Pipeline

# initialize a pipeline for English
p = Pipeline('english')

# a non-empty string to process, which can be a document or a paragraph with multiple sentences
doc_text = '''Hello! This is Trankit.'''

all = p.posdep(doc_text)

print(p._config.__dict__)