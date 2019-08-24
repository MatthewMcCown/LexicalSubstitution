#!/usr/bin/env python
import sys
from lexsub_xml import read_lexsub_xml
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import gensim
import numpy as np
from collections import defaultdict
import string
import json
import requests

def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos):
    possible_synonyms = []
    synset_set = set()
    #retrieve all lexemes for the desired lemma
    retrieve_lemma = wn.lemmas(lemma, pos = pos)
    for item in retrieve_lemma:
        # get the synset for each lexeme
        item_synset = item.synset()
        item_lexemes = item_synset.lemmas() 
        for element in item_lexemes:
            synset_set.add(element.name())
    # Clean up the set and add to the list
    for set_element in synset_set:
        if set_element == lemma:
            pass
        else:
            set_element.replace(' ','_')
            possible_synonyms.append(set_element)
    return possible_synonyms

def wn_frequency_predictor(context):
    frequency_dict = defaultdict(int)
    input_context = context
    target_word = context.lemma
    retrieve_lemma = wn.lemmas(input_context.lemma, pos = input_context.pos)
    for item in retrieve_lemma:
        # get the synset for each lexeme
        item_synset = item.synset()
        item_lexemes = item_synset.lemmas() 
        for element in item_lexemes:
           word_name = element.name()
            # check that the word_name != target_word
           if word_name != target_word:
               word_name.replace(' ','_') # replace any spaces
               frequency_dict[word_name]+= element.count()
    best_synonym = max(frequency_dict, key=frequency_dict.get)
    return best_synonym

def wn_simple_lesk_predictor(context):
    stop_words = stopwords.words('english')
    target_word = context.lemma
    target_word_pos = context.pos
    overlap_dict = defaultdict(int)
    #retrieve all lexemes for the desired lemma
    retrieve_lemma = wn.lemmas(target_word, pos = target_word_pos)
    # Concatenate the left and right contexts of the target word
    total_context = []
    total_context.extend(context.left_context)
    total_context.extend(context.right_context) 
    for lexeme in retrieve_lemma:
        candidate_synset_elements = list()
        synset = lexeme.synset() 
        synset_definition = synset.definition()
        # tokenize definition, remove stop words and add to candidate synset list
        # this also checks that each word != the target word
        for word in tokenize(synset_definition):
            if word not in stop_words:
                if word != target_word:
                    word.replace(' ','_')
                    candidate_synset_elements.append(word)
        # pull the examples of each candidate synset
        synset_examples = synset.examples()
        # tokenize examples, remove stop words and add to candidate synset list
        # this also checks that each word != the target word        
        for sent in synset_examples:
            for word in tokenize(sent):
                if word not in stop_words:
                    if word != target_word:
                        word.replace(' ','_')
                        candidate_synset_elements.append(word)
        # Add the hypernyms and their definitions
        synset_hypernyms = synset.hypernyms() 
        for hypernym in synset_hypernyms:
            hypernym_definition = hypernym.definition()
            for word in tokenize(hypernym_definition):
                if word not in stop_words:
                    if word != target_word:
                        word.replace(' ','_')
                        candidate_synset_elements.append(word)
            hypernym_examples = hypernym.examples()
            for sent in hypernym_examples:
                for word in tokenize(sent):
                    if word not in stop_words:
                        if word != target_word:
                            word.replace(' ','_')
                            candidate_synset_elements.append(word)
        candidate_synset_elements = list(set(candidate_synset_elements)) # remove duplicative entries
        # find the total overlap between the expanded synset and the context of target word
        overlap_count = 0
        for word in total_context:
            if word in candidate_synset_elements:
                overlap_count += 1
        overlap_dict[synset.name()] = overlap_count
    maxOverlap = max(overlap_dict.values())
    tie_test = [k for k, v in overlap_dict.items() if v == maxOverlap]
    # If there is no tie for max overlap:
    if len(tie_test) == 1:
        best_synset = max(overlap_dict, key=overlap_dict.get)
        lemma_dict = defaultdict(int)
        for lemma in wn.synset(best_synset).lemmas():
            if lemma.name() != target_word:
                lemma_dict[lemma.name()] = lemma.count()
        if len(lemma_dict) == 0:
            pass
        else:
            best_synonym_no_tie = max(lemma_dict, key=lemma_dict.get)
            return best_synonym_no_tie
    # If there is a tie for max overlap:
    elif len(tie_test) > 1:
        synset_tie_dict = defaultdict(int)
        for lexeme in retrieve_lemma:
            synset = lexeme.synset()
            for lemma in synset.lemmas():
                if lemma.name() == target_word:
                    synset_tie_dict[synset.name()] += 1
        ordered_synset_tie_list = sorted(synset_tie_dict, key= synset_tie_dict.get,reverse= True)
        for entry in ordered_synset_tie_list:
            most_frequent_lemma_dict = defaultdict(int)
            for lemma in wn.synset(entry).lemmas():
                if lemma.name() != target_word:
                    most_frequent_lemma_dict[lemma.name()] += lemma.count()
            if len(most_frequent_lemma_dict) > 0:
                best_synonym_tie = max(most_frequent_lemma_dict, key=most_frequent_lemma_dict.get)
                return best_synonym_tie

class Word2VecSubst(object):
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def predict_nearest(self,context):
        # part 4
        self.target_word = context.lemma
        self.target_pos = context.pos
        self.candidate_synonyms = get_candidates(self.target_word,self.target_pos)
        self.similarity_dict = defaultdict(int)
        for synonym in self.candidate_synonyms:
            try:
                self.similarity_dict[synonym] = self.model.similarity(self.target_word, synonym)
            except:
                pass # ignore words that are not in the vocabulary 
        self.best_synonym = max(self.similarity_dict, key=self.similarity_dict.get)
        return self.best_synonym

    def predict_nearest_with_context(self, context): 
        stop_words = stopwords.words('english')
        self.target_word = context.lemma
        self.target_pos = context.pos
        self.new_left_context = []
        self.new_right_context = []
        self.new_total_context = []
        self.vectors_to_sum = []
        # Construct the context of +/- 5 words from the target word
        # First, we need to remove stop words and punctuation
        for word in context.left_context:
            if word not in stop_words:
                if word.isalpha():
                    self.new_left_context.append(word)
        for word in context.right_context:
            if word not in stop_words:
                if word.isalpha():
                    self.new_right_context.append(word)
        # The new context will have a radius of 5 words
        # Need to test the length of the cleaned up left and right contexts
        self.new_left_context_length = len(self.new_left_context)
        self.new_right_context_length = len(self.new_right_context)
        if self.new_left_context_length <= 5:
            self.new_total_context.extend(self.new_left_context)
        elif self.new_left_context_length > 5:
            self.new_total_context.extend(self.new_left_context[:self.new_left_context_length-6:-1])
        self.new_total_context.append(self.target_word)
        if self.new_right_context_length <= 5:
            self.new_total_context.extend(self.new_right_context)
        elif self.new_right_context_length > 5:
            self.new_total_context.extend(self.new_right_context[:5:1])
        # Convert the context words into a single vector by
        # summing the constituent vectors
        for word in self.new_total_context:
            try:
                self.vectors_to_sum.append(self.model.wv[word])
            except:
                pass
        self.context_vector = sum(self.vectors_to_sum)
        # Get the candidate synonyms 
        self.candidate_synonyms = get_candidates(self.target_word,self.target_pos)
        self.similarity_dict = defaultdict(int)
        # Build a dict of synonyms : vectors
        for synonym in self.candidate_synonyms:
            try:
                self.similarity_dict[synonym] = self.model.wv[synonym]
            except:
                pass
        self.similarity_score_dict = defaultdict(int)
        # Compute the difference between the context vector and the vectors of the synonyms
        for key, value in self.similarity_dict.items():
            self.similarity_score_dict[key] = np.linalg.norm(self.context_vector - value)
        # Select the synonym vector with the lowest L2-Norm
        self.best_synonym = min(self.similarity_score_dict, key=self.similarity_score_dict.get)
        return self.best_synonym

if __name__=="__main__":

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        prediction = predictor.predict_nearest_with_context(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
