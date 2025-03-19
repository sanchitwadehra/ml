import nltk
from nltk import word_tokenize, pos_tag
from nltk.chunk import RegexpParser
from nltk.tree import Tree

# Download necessary resources
try:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
except Exception as e:
    print(f"Error downloading resources: {e}")

def analyze_text(text):
    # Step 1: Tokenize the text
    tokens = word_tokenize(text)

    # Step 2: Perform PoS tagging
    pos_tags = pos_tag(tokens)
    print("PoS Tags:", pos_tags)

    # Step 3: Define chunking grammar rules
    grammar = """
        NP: {<DT>?<JJ>*<NN>}         # Noun Phrase
        VP: {<VB.*><NP|PP|CLAUSE>+$} # Verb Phrase
        PP: {<IN><NP>}               # Prepositional Phrase
    """

    # Step 4: Apply chunking
    chunk_parser = RegexpParser(grammar)
    chunk_tree = chunk_parser.parse(pos_tags)

    # Step 5: Print chunks
    print("\nIdentified Chunks:")
    for subtree in chunk_tree.subtrees():
        if subtree.label() != 'S':  # Skip the root node
            print(f"{subtree.label()}: {' '.join([token for token, tag in subtree.leaves()])}")

    # Step 6: Print tree structure
    print("\nParse Tree Structure:")
    Tree.fromstring(str(chunk_tree)).pretty_print()

if __name__ == "__main__":
    # Sample text
    text = "The quick brown fox jumps over the lazy dog."
    print("Input text:", text, "\n")
    analyze_text(text)
