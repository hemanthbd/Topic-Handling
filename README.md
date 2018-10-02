# Topic-Handling

Program to Perform Topic Handling using Latent Dirichlet Allocation (LDA).

## Files Description

* topic_modeling_data.json - Input Dataset consisting of 1872 documents with corresponding ids.

Sample input :
```
 {"_id": "abcdef", "text": "This is an example document."}
```
* Topic_mod_main.py - Main File

Description :
```
 This Program takes in the input .json file and writes to an output file, "output_tm_main.json", with 5 
 topics for each id, corresponding to each input document, and the topics are arranged in descending 
 order (Highest being most similar to the document, and Lowest being least similar) of Scores.
 Dictionary used for the LDA Model comprises of all the tokenized words from the input file, as well as the Corpus.
 The LDA model from Gensim used here is LDAMultiCore, as it can parallelize it's computation, hence give a faster 
 output than the regular LDA.
```
* output_tm_main.json - Output File of Topic_mod_main.py

Sample Output :
```
 {"_id": "abcdef", "topics": ["topic1", "topic2", "topic3", "topic4", "topic5”]}
```
* Topic_mod_val.py - Validation File

Description :
```
 The Primary function of this program is to test the Output from the main "Topic_mod_main.py" program.
 This Program takes in the input .json file and writes to an output file, "output_tm_val.json", with 5 
 topics for each id, corresponding to each input document, and the topics are arranged in descending 
 order (Highest being most similar to the document, and Lowest being least similar) of Scores.
 However, here, an iteration is run across all the documents, and the Dictionary is chosen for each 
 document comprising of words from only that document, as well as the Corpus, and thus the Output Topics 
 would be highly unique only to that particular document. 
 The LDA model from Gensim used here is the standard LDA model, just so as to have a comparision between 
 this and the LDAMultiCore model in the main file.
```

* output_tm_val.json - Output File of Topic_mod_val.py

Sample Output :
```
 {"_id": "abcdef", "topics": ["topic1", "topic2", "topic3", "topic4", "topic5”]}
```
## Implemetation

* The files were run on 
```
 * Intel(R) Core(TM) i7-4700MQ @ 2.4 GHz with 16-GB RAM 
 * NVIDIA GeForce GTX 77
 * Linux x64 bit OS
```
* Python Package - Python3

* To run file on a typic Linux Environment
```
 ~$ cd <to location of your files>
 :~/location/of/your/files/$ ./python3 Topic_mod_main.py
 The Output file "output_tm_main.json" will be downloaded in the location where your files are.
 :~/location/of/your/files/$ ./python3 Topic_mod_val.py
 The Output file "output_tm_val.json" will be downloaded in the location where your files are.
```

### Packages

What things you need to install the software and how to install them

* [Gensim](https://github.com/RaRe-Technologies/gensim) - Package used for Text Modeling
```
 conda install -c anaconda gensim 
```
* [NLTK](https://maven.apache.org/) - Package used for Text 'Cleaning'/ Pre-processing
```
 conda install -c anaconda nltk 
 ```
* [JSON](https://docs.python.org/3.5/library/json.html) - Package used to Read/Write into JSON files
```
import json
```
* [RE](https://docs.python.org/3.5/library/re.html) - Package used for String expression matching
```
import re
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://opensource.org/licenses/MIT) file for details

## Acknowledgement 

[RedMarlin](https://www.redmarlin.ai/) - For the Coding Challenge
