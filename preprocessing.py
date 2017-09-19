## coding=utf8
from nltk.tokenize import word_tokenize, sent_tokenize
from pattern_tokenizer import pattern_tokenize
from xml.etree import ElementTree


#preprocessing
# met nltk_tokenize
def preprocess(text, split_sentences=True, return_original=False, lowercase=True):
	if return_original: 
		return text
	else: 
		if split_sentences:
			sentences = sent_tokenize(text)
			processed = list()
			for splitted_sentence in sentences:
				for token in word_tokenize(splitted_sentence):
					processed.append(token)
		else:
			processed = word_tokenize(text)

		if lowercase:
			processed = [token.lower() for token in processed] 

		return processed

text = """Autism is a neurodevelopmental disorder characterized by impaired social interaction, impaired verbal and non-verbal communication, and restricted and 
repetitive behavior. Parents usually notice signs in the first two years of their child's life. These signs often develop gradually, though some children with 
autism reach their developmental milestones at a normal pace and then regress. The diagnostic criteria require that symptoms become apparent in early 
childhood, typically before age three.
Autism is caused by a combination of genetic and environmental factors. Some cases are strongly associated with certain infections during pregnancy 
including rubella and use of alcohol or cocaine. Controversies surround other proposed environmental causes;for example the vaccine hypotheses, 
which have been disproven. Autism affects information processing in the brain by altering how nerve cells and their synapses connect and organize; how this 
occurs is not well understood.In the DSM V, autism is included within the autism spectrum (ASDs), as is Asperger syndrome, which lacks delays in cognitive 
evelopment and language, and pervasive developmental disorder, not otherwise specified (commonly abbreviated as PDD-NOS), which was diagnosed when the full set 
of criteria for autism or Asperger syndrome were not met.
Early speech or behavioral interventions can help children with autism gain self-care, social, and communication skills. Although there is no known cure,
there have been reported cases of children who recovered. Not many children with autism live independently after reaching adulthood, though some become 
successful. An autistic culture has developed, with some individuals seeking a cure and others believing autism should be accepted as a difference and not 
treated as a disorder.
Globally, autism is estimated to affect 24.8 million people as of 2015. As of 2010, the number of people affected is estimated at about 1-2 per 1,000 worldwide. 
It occurs four to five times more often in boys than girls. About 1.5%% of children in the United States (one in 68) are diagnosed with ASD as of 2014, 
a 30%% increase from one in 88 in 2012. The rate of autism among adults aged 18 years and over in the United Kingdom is 1.1%%. The number of people diagnosed 
has been increasing dramatically since the 1980s, partly due to changes in diagnostic practice; the question of whether actual rates have increased is unresolved."""

#text = text.encode("ascii","ignore")

# tokenized_text = preprocess(text)
# print(tokenized_text)


#preprocessing
# met pattern_tokenizer
tokenized_text2 = []
for tokenized_sentence in pattern_tokenize(text):
	tokenized_text2 += tokenized_sentence.lower().split()
print(tokenized_text2)


