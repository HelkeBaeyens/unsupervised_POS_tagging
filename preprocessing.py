#based on: https://de.dariah.eu/tatom/working_with_text.html
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from os import listdir
from os.path import isfile, join
import sys

if len(sys.argv) > 1:
	mypath = sys.argv[1]
else: 
	mypath = "data"

filenames = [mypath+"/"+f for f in listdir(mypath) if isfile(join(mypath, f))]

vectorizer = CountVectorizer(input='filename', strip_accents='unicode')

dtm = vectorizer.fit_transform(filenames) # sparse matrix
vocab = vectorizer.get_feature_names()
print(vocab)
dtm = dtm.toarray()  # to convert to a regular array
vocab = np.array(vocab)
# output = regular arrays

