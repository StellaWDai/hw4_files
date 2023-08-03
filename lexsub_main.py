#!/usr/bin/env python
import sys
import string
from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 

from typing import List

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    res = []
    for i in wn.lemmas(lemma,pos):
        for j in i.synset().lemmas():
            if j.name() != lemma:
                exp = j.name().replace('_', ' ')
                res.append(exp)
    return set(res) 

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    res = dict()
    for i in wn.lemmas(context.lemma,context.pos):
        for j in i.synset().lemmas():
            if j.name() != context.lemma:
                exp = j.name().replace('_', ' ')
                res[exp] = res.get(exp,0) + j.count()
    return max(res,key=res.get) # replace for part 2

def wn_simple_lesk_predictor(context : Context) -> str:
    stop_words = set(stopwords.words('english'))
    sentence = set(context.left_context+context.right_context)
    overlap_len={}
    #get the context without stop words
    sentence = sentence-stop_words
    lexemes = wn.lemmas(context.lemma,context.pos)
    for lemma in lexemes:
        syn = lemma.synset()
        #get the definition of synsets
        definition = tokenize(syn.definition())
        #add all examples of synsets to definition
        for example in syn.examples():
            definition +=tokenize(example)
        #add definitions and examples of hypernyms of synsets
        for hyper in syn.hypernyms():
            definition +=tokenize(hyper.definition())
            for hyper_ex in hyper.examples():
                definition +=tokenize(hyper_ex)
        #get the definitions without stop words
        definition = set(definition)-stop_words
        #get the overlap score
        overlap_len[lemma] = len(definition & sentence)
    freq={}
    res=''
    #get the synset with the most overlap
    for lemma in lexemes:
        syn = lemma.synset() 
        if lemma in overlap_len and overlap_len[lemma]==max(overlap_len.values()):
            for lem in syn.lemmas():
                name = lem.name()
                freq[name] = lem.count()
    #remove the target from the dictionary
    if context.lemma in freq:
         del freq[context.lemma]
    #sort lexeme by its frequency
    ranked_names = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
    ranked_keys = list(ranked_names.keys())
    return ranked_keys[0].replace('_', ' ') #replace for part 3    

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        candidates = get_candidates(context.lemma,context.pos)
        res = ''
        score = 0
        for candidate in candidates:
            try:
                if self.model.similarity(candidate,context.lemma)>=score:
                    score = self.model.similarity(candidate,context.lemma)
                    res = candidate
            except:
                pass
        return res # replace for part 4


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        #get candidate symnonyms
        candidates = get_candidates(context.lemma,context.pos)
        #BERT
        full=''
        for i in context.left_context:
            full+=i
        full = full+' '+'[MASK]'+' '
        for i in context.right_context:
            full+=i
        input_toks = self.tokenizer.encode(full)
        tokenized_input = self.tokenizer.convert_ids_to_tokens(input_toks)
        index = tokenized_input.index('[MASK]')
        input_mat = np.array(input_toks).reshape((1,-1))  # get a 1 x len(input_toks) matrix
        outputs = self.model.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][index])[::-1]
        words = self.tokenizer.convert_ids_to_tokens(best_words)
        res=''
        for word in words:
            word = word.replace('_',' ')
            if word in candidates:
                res = word
        return res # replace for part 5

    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)
    bert_predictor = BertPredictor()

    for context in read_lexsub_xml(sys.argv[1]):
        print(context)  # useful for debugging
        #prediction = smurf_predictor(context) 
        #prediction = wn_frequency_predictor(context)
        #prediction = wn_simple_lesk_predictor(context)
        #prediction = predictor.predict_nearest(context)
        prediction = bert_predictor.predict(context)

        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))