
# coding: utf-8

# ## Imports

# In[1]:


import nltk
import json
from flask import Flask, g, Response, request, jsonify
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import SequentialBackoffTagger
from neo4j.v1 import GraphDatabase


# In[2]:


app = Flask(__name__, static_url_path='/static/')


# ## Connecting to Neo4j

# In[3]:


driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
db = driver.session()


# ## Tagger

# In[4]:


class PartsOfQueryTagger(SequentialBackoffTagger):
    
    def __init__(self, *args, **kwargs):
        SequentialBackoffTagger.__init__(self, *args, **kwargs)
    
    def choose_tag(self, tokens, index, history):
        results = db.run('MATCH (n {name: {name}}) return n', {"name":tokens[index]})
        return results;


# In[5]:


tagger = PartsOfQueryTagger()


# In[6]:


tagged_words = tagger.tag(word_tokenize("create a concept node with name as MEANStack"))


# In[7]:


for word, results in tagged_words:
    for record in results:
        print(record)


# ## Feature Extraction

# In[8]:


def extract_features(sentence):
    """
    Extracts features from a sentence
    
    sentence: Inputs a sentence in natural language
    Returns: a dict of feature, indicating a presence or absence of certain features
    """
    tokenized_words = [word.lower() for word in word_tokenize(sentence)]
    features = ['node', 'nodes', 'relations', 'relationships', 'csv', 'build', 'make', 'match', 'find', 'fetch', 'create', 'get', 'number', 'count', 'relation']
    word_dict = {}
    for feature in features:
        if feature in tokenized_words:
            word_dict[feature] = +1
        else:
            word_dict[feature] = -1
    return word_dict


# In[9]:


import random
datasets = ['createNodes', 'createRelations', 'getCountOfNodes', 'getNodes', 'createNodesFromCsv', 'createRelationsFromCsv']
raw_dataset = []
for file in datasets:
    with open('data/'+file) as f:
        raw_dataset += [(extract_features(question), file) for question in f]
random.shuffle(raw_dataset)
size = int(len(raw_dataset) * 0.5)
train_set, test_set = raw_dataset[:size], raw_dataset[size:]
queryClassifier = nltk.NaiveBayesClassifier.train(train_set)


# In[12]:


@app.route('/classify', methods=['POST'])
def classify_query():
    print(request.json);
    return jsonify({'intent': queryClassifier.classify(extract_features(request.json["query"]))});


# In[11]:


# classify_query('create relations from csv file')


# In[ ]:


if __name__ == '__main__':
    app.run(port=8082)

