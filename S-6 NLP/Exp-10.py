import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def chunk_text(text):
    # Tokenize and POS tag
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    
    # Define grammar for noun phrases
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    chunk_parser = nltk.RegexpParser(grammar)
    
    # Apply chunking
    chunk_tree = chunk_parser.parse(pos_tags)
    
    # Print results
    print("\nChunked Structure:")
    print(chunk_tree)

# Example usage
sample_text = "The quick brown fox jumps over the lazy dog."
chunk_text(sample_text)
