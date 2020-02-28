import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

import numpy as np
import nltk
from nltk.corpus import wordnet as wn


def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
    
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


def doc_to_synsets(doc):
    """
    Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.

    Args:
        doc: string to be converted

    Returns:
        list of synsets

    Example:
        doc_to_synsets('Fish are nvqjp friends.')
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    """
    

    words = nltk.word_tokenize(doc)
    
    pos_list = nltk.pos_tag(words)
    synsets = []
    
    for word, tag in pos_list:
        newtag = convert_tag(tag)
        syn = wn.synsets(word, pos=newtag)
        if not syn:
            continue
        else:
            synsets.append(syn[0])
    
    return synsets


def similarity_score(s1, s2):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        synsets1 = doc_to_synsets('I like cats')
        synsets2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
    """

    synsets1 = s1
    synsets2 = s2
    
    max_sim = []
    for syn1 in synsets1:
        similarity = []
        for syn2 in synsets2:
            s = syn1.path_similarity(syn2)
            if s != None:
                similarity.append(s)
            else:
                similarity.append(0)
                
        if not similarity:
            continue

        max_sim.append(max(similarity))
    
    return np.mean(max_sim)


def document_path_similarity(doc1, doc2):
    """Finds the symmetrical similarity between doc1 and doc2"""

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)

    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2
