#https://de.dariah.eu/tatom/working_with_text.html
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sys import exit

filenames = ['data/austen-brontë/Austen_Emma.txt',
             'data/austen-brontë/Austen_Pride.txt',
             'data/austen-brontë/Austen_Sense.txt',
             'data/austen-brontë/CBronte_Jane.txt',
              'data/austen-brontë/CBronte_Professor.txt',
              'data/austen-brontë/CBronte_Villette.txt']
vectorizer = CountVectorizer(input='filename')

dtm = vectorizer.fit_transform(filenames)  # a sparse matrix - DocumentTermMatrix - puts data CountVectorizer in a matrix
vocab = vectorizer.get_feature_names()  # a list
#print(dtm)
#print(vocab)
# for reference, note the current class of `dtm`
#print(type(dtm))
dtm = dtm.toarray()  # convert to a regular array
vocab = np.array(vocab)
#print(dtm)

# use the standard Python list method index(...)
# list(vocab) or vocab.tolist() will take vocab (an array) and return a list
# first index is book, second is word
house_idx = list(vocab).index('house')
dtm[0, house_idx]
# using NumPy indexing will be more natural for many
# dtm[0, vocab == 'house']
# a_test_array = [1,55,8]
# print(dtm)
# print(type(dtm))
# a_test_matrix = [[1,55,8],
#                    [8,8,8],
#                    [1,1,1]]
# r = a_test_matrix[2][2]
# print(r) 
# r = a_test_matrix[0,0]  -----> does not work for plain array in contrary to numpy.ndarray
# print(r) 

# r = a_test_array[1] 
55 # just a number
#print(house_idx)
# # print(dtm[0, house_idx])
# house = dtm[0:,vocab == 'house'] 
# print (house)
# for row in dtm:
# 	print (row[vocab == 'house'])
# for i in range(6):
# 	print(dtm[i,vocab == 'house'])
# "by hand Euclidian distance"
# n, _ = dtm.shape
# dist = np.zeros((n, n))
# for i in range(n):
# 	for j in range(n):
#    		x, y = dtm[i, :], dtm[j, :]
#    		dist[i, j] = np.sqrt(np.sum((x - y)**2))
# print(np.round(dist, 1))

# sklearn
# print(len(vocab)) # dimensions
# from sklearn.metrics.pairwise import euclidean_distances
# dist = euclidean_distances(dtm)
# print(np.round(dist, 1))

print(len(vocab)) # dimensions
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity (dtm)
print(similarity)







