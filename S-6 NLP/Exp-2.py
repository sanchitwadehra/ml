# Experiment 2: Stemming and Lemmatization
# Aim: To implement and understand the concepts of Stemming and Lemmatization in NLP

# Import required libraries
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer, LancasterStemmer, RegexpStemmer, SnowballStemmer
from nltk.corpus import stopwords

# Download required NLTK data
print("Downloading required NLTK packages...")
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
print("Downloads completed.\n")

def demonstrate_basic_stemming_lemmatization():
    print("\n=== Basic Stemming and Lemmatization ===")
    # Sample text
    text = "The quick brown foxes are jumping over the lazy dogs. They also love running in the fields."
    print("Original text:", text)

    # Tokenization
    tokens = nltk.word_tokenize(text)
    print("\nTokenized words:", tokens)

    # Stemming using Porter Stemmer
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in tokens]
    print("\nStemmed Words:", stemmed_words)

    # Basic Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
    print("\nLemmatized Words:", lemmatized_words)

def demonstrate_advanced_lemmatization():
    print("\n=== Advanced Lemmatization with POS Tags ===")
    
    def lemmatize_with_pos(text):
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = []
        
        for word, pos in pos_tags:
            if pos.startswith('V'):  # Verb
                lemmatized_words.append(lemmatizer.lemmatize(word, pos='v'))
            elif pos.startswith('J'):  # Adjective
                lemmatized_words.append(lemmatizer.lemmatize(word, pos='a'))
            elif pos.startswith('R'):  # Adverb
                lemmatized_words.append(lemmatizer.lemmatize(word, pos='r'))
            else:  # Noun
                lemmatized_words.append(lemmatizer.lemmatize(word, pos='n'))
        
        return lemmatized_words

    # Test text
    text = "The children were playing happily. They are running faster than before."
    print("Original text:", text)
    advanced_lemmatized = lemmatize_with_pos(text)
    print("\nAdvanced Lemmatized Words:", advanced_lemmatized)

def compare_stemmers():
    print("\n=== Comparing Different Stemmers ===")
    
    # Initialize stemmers
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    regexp = RegexpStemmer('ing$|s$|ed$', min=4)
    snowball = SnowballStemmer('english')

    # Sample words
    words = ['running', 'cats', 'fishing', 'trouble', 'connection', 
             'organization', 'truthful', 'probabilities']
    print("Original Words:", words)

    # Apply different stemmers
    porter_stemmed = [porter.stem(word) for word in words]
    lancaster_stemmed = [lancaster.stem(word) for word in words]
    regexp_stemmed = [regexp.stem(word) for word in words]
    snowball_stemmed = [snowball.stem(word) for word in words]

    # Print results
    print("\nPorter Stemmer:", porter_stemmed)
    print("Lancaster Stemmer:", lancaster_stemmed)
    print("Regexp Stemmer:", regexp_stemmed)
    print("Snowball Stemmer:", snowball_stemmed)

def main():
    print("=== Experiment 2: Stemming and Lemmatization ===")
    
    # Run all demonstrations
    demonstrate_basic_stemming_lemmatization()
    demonstrate_advanced_lemmatization()
    compare_stemmers()
    
    print("\n=== Learning Outcomes ===")
    print("1. Understood the difference between Stemming and Lemmatization")
    print("2. Implemented various stemming algorithms")
    print("3. Applied advanced lemmatization with part-of-speech tagging")
    print("4. Compared different stemming techniques")

if __name__ == "__main__":
    main() 