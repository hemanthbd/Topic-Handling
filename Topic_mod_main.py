# Program to perform Topic-Modeling for n Documents using LDA (Dictionary is the Entire Corpus)

import json
from pprint import pprint
import numpy as np
import string
import nltk
import re
import gensim
import matplotlib.pyplot as plt

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from gensim import corpora, models


# Initialize the Tokenizer
tokenizer = RegexpTokenizer(r'\w+')

#Download 'Stop-Words' package from NTLK
nltk.download('stopwords')

#Download 'Wordnet' package from NTLK
nltk.download('wordnet')

# Use Stop-words in English & German , as Dataset has both English & German Documents
stopwords = stopwords.words('english','german')

#Initialize the Stemmer 
stem = PorterStemmer()

#Initilaize the Lemmatizer
lem = nltk.wordnet.WordNetLemmatizer()


doc_list=[]
raw = []
only_raw =[]
tokens = []
new = []
final = []
data = []
results = []

print("\nOpening Input json file")
# Open the Input .json file which contains the documents
with open('topic_modeling_data.json') as f:
	dic = [json.loads(line) for line in f]
		
#pprint(your_json[0]['text'])

#Loop to store the 'text' column of the file
for i in range(0,len(dic)):
	doc_list.append(dic[i]['text'].translate(string.punctuation))

# Loop to convert into Lowercase
for i in range(0,len(doc_list)):
	raw.append(doc_list[i].lower())
print('\nTotal Number of Documents',len(raw))

# Loop to remove Numbers and few symbols
for i in range(0,len(raw)):
	only_raw.append(re.sub('[0-9]+', '',raw[i]))
print('Removed Numbers/Symbols')

# Loop to convert each word into a token
for i in range(0,len(only_raw)):
	tokens.append(tokenizer.tokenize(only_raw[i]))
print('Tokenized the Data')

# Loop to collect all the tokens except stopwords
for i in range(0,len(tokens)):
	new.append([p for p in tokens[i] if not p in stopwords])
print('Tokenized Data - Stop-Words')

#Loop to 'Lemmatize' each word, ie, Stemm the words, but according to context.
for i in range(0,len(new)):
	final.append([lem.lemmatize(p) for p in new[i]])
print('Data Stemmed')

imp1=[]
results = []
topics = []
total = []
tsc=[]
total_score=[]

# Calculate Dictionary for the Entire Dataset of Documents
dictionary = corpora.Dictionary(final)
print('\nDictionary Set')
# Perform K-Means clustering or Bag of Words clustering for Entire set of Documents, and create a Corpus.
corpus = [dictionary.doc2bow(text) for text in final]
print('Corpus Set')

# Create an LDA MultiCoreModel and pass the Corpus and Dictionary to it, and choose 5 topics with 40 passes and workers=3 (Parallelize)
ldamodel = gensim.models.LdaMulticore(corpus, num_topics=30, id2word = dictionary, passes=40, minimum_probability=0, workers=3)
print('\n LDA Model Defined')
print('\n')

# Loop through every Document
for i in range(0,len(final)):
	# Sort the Topics by Scores in Descending Order
	for j, score in sorted(ldamodel[corpus[i]], key=lambda x: -1*x[1]):
		
		# Topics with 4 words each		
		x = ldamodel.print_topic(j,4)
		
		topics.append(x)
		
		#Append each Topic Score as well.
		tsc.append(score)
	total.append(topics)
	total_score.append(tsc)
	topics =[]
	tsc=[]

	print("Topics collected for the {}st Document".format(i))


total_score= np.asarray(total_score)
#print('\n Scores for each Topic per Document', total_score)

words=[]
words1 = []

# Iterate through every topic , append first 5 topics based on Score, and remove Numbers (which are the Probab. Dist of each word in the corpus) and any other symbol like spaces or '*'
for i in range(0,len(final)):
	for j in range(0,5):
		words1.append(re.sub (r'([^a-zA-Z ]+?)', '', total[i][j]))
	words.append(words1)
	words1 = []


d2 = []

print("\n Writing into Json file")

# Store every topic in a Dictionary format to export into a .json file
for i in range(0,len(final)):  
	d = {'_id': dic[i]['_id'],'topics': [words[i][0], words[i][1], words[i][2],words[i][3],words[i][4]]} 
	d2.append(d)

# Write into an empty .json file with format: {"_id:":"topics":["topic1","topic2","topic3","topic4","topic5"]} with in Order of Decreasing Scores
with open('tr3.json', 'w') as outfile:  
    for item in d2:
            x = json.dumps(item,sort_keys=True)
            outfile.write(x+'\n')

print('\n Task End')
