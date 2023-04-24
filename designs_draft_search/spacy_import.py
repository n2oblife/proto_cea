# import spacy

# # Load English tokenizer, tagger, parser and NER
# nlp = spacy.load("en_core_web_sm")

# print('\n--------------------------------------------- \n')
# print(nlp.config.keys)
# print('\n--------------------------------------------- \n')
# print(nlp.components[2])
# print('\n--------------------------------------------- \n')
# print(nlp.pipeline[2])
# print('\n--------------------------------------------- \n')
# print(nlp.get_pipe("parser"))

# # Process whole documents
# text = ("When Sebastian Thrun started working on self-driving cars at "
#         "Google in 2007, few people outside of the company took him "
#         "seriously. “I can tell you very senior CEOs of major American "
#         "car companies would shake my hand and turn away because I wasn’t "
#         "worth talking to,” said Thrun, in an interview with Recode earlier "
#         "this week.")
# doc = nlp(text)

# # Analyze syntax
# print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
# print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# # Find named entities, phrases and concepts
# for entity in doc.ents:
#     print(entity.text, entity.label_)

import spacy
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL

config = {
   "moves": None,
   "update_with_oracle_cut_size": 100,
   "learn_tokens": False,
   "min_action_freq": 30,
   "model": DEFAULT_PARSER_MODEL,
}

nlp = spacy.load("en_core_web_sm")
nlp.remove_pipe("tagger")
nlp.remove_pipe("lemmatizer")
nlp.remove_pipe("attribute_ruler")
nlp.remove_pipe("ner")
nlp.remove_pipe("tok2vec")
nlp.remove_pipe("senter")


print(nlp.config)