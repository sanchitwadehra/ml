import nltk
import numpy as np
from nltk.corpus import brown
from collections import defaultdict

# Download required NLTK data
nltk.download('brown')
nltk.download('universal_tagset')

class HMM_POS_Tagger:
    def __init__(self):
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.emission_probs = defaultdict(lambda: defaultdict(float))
        self.tag_counts = defaultdict(float)
        
    def train(self, tagged_sentences):
        # Count transitions and emissions
        for sentence in tagged_sentences:
            prev_tag = '<START>'
            for word, tag in sentence:
                # Count tag transitions
                self.transition_probs[prev_tag][tag] += 1
                # Count word emissions
                self.emission_probs[tag][word] += 1
                self.tag_counts[tag] += 1
                prev_tag = tag
        
        # Normalize probabilities
        for prev_tag in self.transition_probs:
            total = sum(self.transition_probs[prev_tag].values())
            for tag in self.transition_probs[prev_tag]:
                self.transition_probs[prev_tag][tag] /= total
                
        for tag in self.emission_probs:
            total = self.tag_counts[tag]
            for word in self.emission_probs[tag]:
                self.emission_probs[tag][word] /= total
    
    def viterbi(self, sentence):
        V = [{}]
        path = {}
        
        # Initialize
        tags = set(self.tag_counts.keys())
        for tag in tags:
            V[0][tag] = self.transition_probs['<START>'][tag] * self.emission_probs[tag].get(sentence[0], 1e-10)
            path[tag] = [tag]
            
        # Run Viterbi
        for t in range(1, len(sentence)):
            V.append({})
            newpath = {}
            
            for curr_tag in tags:
                (prob, state) = max((V[t-1][prev_tag] * 
                                   self.transition_probs[prev_tag].get(curr_tag, 1e-10) * 
                                   self.emission_probs[curr_tag].get(sentence[t], 1e-10), prev_tag) 
                                   for prev_tag in tags)
                V[t][curr_tag] = prob
                newpath[curr_tag] = path[state] + [curr_tag]
            path = newpath
            
        # Find best path
        (prob, state) = max((V[len(sentence) - 1][tag], tag) for tag in tags)
        return path[state]

def main():
    # Get training data
    tagged_sents = brown.tagged_sents(tagset='universal')[:10000]
    
    # Train the model
    tagger = HMM_POS_Tagger()
    tagger.train(tagged_sents)
    
    # Test sentence
    test_sentence = "The cat sat on the mat".split()
    tags = tagger.viterbi(test_sentence)
    
    # Print results
    print("\nPOS Tags for:", test_sentence)
    print(list(zip(test_sentence, tags)))

if __name__ == "__main__":
    main()
