# Program to perform Topic-Modeling on 'n' Documents using LDA (Dictionary is each document individually by itself)
# This Program is like a Validation Set for the Main Program.

import json
from pprint import pprint
import numpy as np
import string
import nltk
import re
import gensim

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
print('Original',len(raw))

# Loop to remove Numbers and few symbols
for i in range(0,len(raw)):
	only_raw.append(re.sub('[0-9]+', '',raw[i]))

# Loop to convert each word into a token
for i in range(0,len(only_raw)):
	tokens.append(tokenizer.tokenize(only_raw[i]))

# Loop to sotre all the tokens eacept those present in stopwords
for i in range(0,len(tokens)):
	new.append([p for p in tokens[i] if not p in stopwords])

#Loop to 'Lemmatize' each word, ie, Stemm the words, but according to context.
for i in range(0,len(new)):
	final.append([lem.lemmatize(p) for p in new[i]])


imp1=[]
results = []
topics = []
total = []

# Loop through every sentence/document
for i in range(0,len(final)):
	t = len(final[i])
	# Split each document into halves and store in a nested list.
	final1 = np.array(final[i])
	if (len(final)%2!=0):
		imp1.append(final1[0:int((t-1)/2)])
		imp1.append(final1[int((t-1)/2):t])
	else:
		imp1.append(final1[0:int((t)/2)])
		imp1.append(final1[int((t)/2):t])

	# Addthe above list corresponding to the 'i'th document to a Dictionary.
	dictionary = corpora.Dictionary(imp1)

    # Perform K-Means clustering or Bag of Words clustering for the 'i'th document/list and create a Corpus.
	corpus = [dictionary.doc2bow(text) for text in imp1]

     # Create an LDA Model and pass the Corpus and DIctionary to it, and choose 5 topics with 40 passes
	ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word = dictionary, passes=40)
	# Iterate through each of the 5 topics having 3 words each, and store the topics for each document
	for j in  ldamodel.show_topics(num_topics=5,num_words=3):
		for m in range(1,len(j),2):
			#print(j[m])
			topics.append(j[m])
	total.append(topics)
	topics=[]

	print('Done',i)

	imp1 = []

words=[]
words1 = []

# Iterate through every topic and remove Numbers (which are the Probab. Dist of each word in the corpus) and any other symbol like spaces or '*'
for i in range(0,len(final)):
	for j in range(0,5):
		words1.append(re.sub (r'([^a-zA-Z ]+?)', '', total[i][j]))
	words.append(words1)
	words1 = []


d2 = []

# Store every topic in a Dictionary format to export into a .json file
for i in range(0,len(final)):  
	d = {'_id': dic[i]['_id'],'topics': [words[i][0], words[i][1], words[i][2],words[i][3],words[i][4]]} 
	d2.append(d)

#print(d2)

# Write into an empty .json file with format: {"_id:":"topics":["topic1","topic2","topic3","topic4","topic5"]}
with open('output_tm_val.json', 'w') as outfile:  
    for item in d2:
            x = json.dumps(item,sort_keys=True)
            outfile.write(x+'\n')
    #json.dump(d2, outfile)


print('done')
