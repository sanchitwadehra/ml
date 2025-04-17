import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def pos_tagger(text):
    # Tokenize and tag
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    
    # Print results
    print("\nPOS Tags:")
    for word, tag in pos_tags:
        print(f"{word}: {tag}")

# Example usage
sample_text = "Natural Language Processing enables machines to understand human language."
pos_tagger(sample_text)
