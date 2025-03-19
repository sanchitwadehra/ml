import nltk
from nltk.corpus import brown
from collections import defaultdict

# Download required NLTK data
nltk.download('brown')
nltk.download('universal_tagset')

class ViterbiPOSTagger:
    def __init__(self):
        self.transitions = defaultdict(lambda: defaultdict(float))
        self.emissions = defaultdict(lambda: defaultdict(float))
        self.tag_counts = defaultdict(int)
        
    def calculate_probabilities(self, training_data):
        # Calculate transition and emission counts
        for sentence in training_data:
            prev_tag = '<START>'
            for word, tag in sentence:
                # Count transitions
                self.transitions[prev_tag][tag] += 1
                # Count emissions
                self.emissions[tag][word.lower()] += 1
                self.tag_counts[tag] += 1
                prev_tag = tag
            
        # Normalize probabilities
        for prev_tag in self.transitions:
            total = sum(self.transitions[prev_tag].values())
            for tag in self.transitions[prev_tag]:
                self.transitions[prev_tag][tag] /= total
        
        for tag in self.emissions:
            total = self.tag_counts[tag]
            for word in self.emissions[tag]:
                self.emissions[tag][word] /= total
    
    def tag_sentence(self, sentence):
        # Initialize Viterbi matrix and backpointers
        V = [{}]
        bp = [{}]
        tags = list(self.tag_counts.keys())
        
        # Initialize first word
        for tag in tags:
            word = sentence[0].lower()
            V[0][tag] = (self.transitions['<START>'].get(tag, 1e-10) * 
                        self.emissions[tag].get(word, 1e-10))
            bp[0][tag] = '<START>'
        
        # Run Viterbi for subsequent words
        for t in range(1, len(sentence)):
            V.append({})
            bp.append({})
            word = sentence[t].lower()
            
            for curr_tag in tags:
                # Find most likely previous tag
                max_prob, best_prev_tag = max(
                    (V[t-1][prev_tag] * 
                     self.transitions[prev_tag].get(curr_tag, 1e-10) * 
                     self.emissions[curr_tag].get(word, 1e-10), prev_tag)
                    for prev_tag in tags
                )
                V[t][curr_tag] = max_prob
                bp[t][curr_tag] = best_prev_tag
        
        # Find best path
        best_tag = max(V[-1], key=V[-1].get)
        best_path = [best_tag]
        
        # Backtrack
        for t in range(len(sentence)-1, 0, -1):
            best_tag = bp[t][best_tag]
            best_path.append(best_tag)
        
        return list(reversed(best_path))

def main():
    # Load training data
    training_data = brown.tagged_sents(tagset='universal')[:5000]
    
    # Initialize and train tagger
    tagger = ViterbiPOSTagger()
    tagger.calculate_probabilities(training_data)
    
    # Test sentences
    test_sentences = [
        "The cat sat on the mat".split(),
        "I love programming in Python".split()
    ]
    
    # Tag and print results
    for sentence in test_sentences:
        tags = tagger.tag_sentence(sentence)
        print("\nSentence:", sentence)
        print("Tagged:", list(zip(sentence, tags)))

if __name__ == "__main__":
    main()
