import nltk
from nltk import word_tokenize, pos_tag
import matplotlib.pyplot as plt
from collections import Counter

def main():
    # Download required resources (downloads only if not already available)
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    # Input text for tagging
    text = "Natural Language Processing is fascinating and powerful."
    
    # Tokenize the text into words
    tokens = word_tokenize(text)
    
    # Apply PoS Tagging to the tokens
    tagged_tokens = pos_tag(tokens)
    
    # Print the tagged tokens
    print("Tagged Tokens:")
    for token, tag in tagged_tokens:
        print(f"{token}: {tag}")
    
    # Optional: Plot the frequency of PoS tags
    tags = [tag for _, tag in tagged_tokens]
    tag_counts = Counter(tags)
    
    plt.bar(tag_counts.keys(), tag_counts.values(), color='skyblue')
    plt.xlabel('PoS Tags')
    plt.ylabel('Frequency')
    plt.title('Frequency of PoS Tags')
    plt.show()

if __name__ == "__main__":
    main()


# Below is a short and concise Python script for POS tagging using NLTK. This code downloads the necessary resources, tokenizes the input text, applies POS tagging, prints the tagged tokens, and also (optionally) plots the frequency distribution of the POS tags.

# ```python:S-6 NLP/Exp-5.py
# import nltk
# from nltk import word_tokenize, pos_tag
# import matplotlib.pyplot as plt
# from collections import Counter

# def main():
#     # Download required resources (downloads only if not already available)
#     nltk.download('punkt')
#     nltk.download('averaged_perceptron_tagger')

#     # Input text for tagging
#     text = "Natural Language Processing is fascinating and powerful."
    
#     # Tokenize the text into words
#     tokens = word_tokenize(text)
    
#     # Apply PoS Tagging to the tokens
#     tagged_tokens = pos_tag(tokens)
    
#     # Print the tagged tokens
#     print("Tagged Tokens:")
#     for token, tag in tagged_tokens:
#         print(f"{token}: {tag}")
    
#     # Optional: Plot the frequency of PoS tags
#     tags = [tag for _, tag in tagged_tokens]
#     tag_counts = Counter(tags)
    
#     plt.bar(tag_counts.keys(), tag_counts.values(), color='skyblue')
#     plt.xlabel('PoS Tags')
#     plt.ylabel('Frequency')
#     plt.title('Frequency of PoS Tags')
#     plt.show()

# if __name__ == "__main__":
#     main()
# ```

# ### Explanation

# 1. **Imports:**
#    - `nltk` is used for tokenization and POS tagging.
#    - `matplotlib.pyplot` and `collections.Counter` are used for optional visualization.
   
# 2. **Resource Downloads:**
#    - The script ensures the required NLTK data (`punkt` and `averaged_perceptron_tagger`) are available.
   
# 3. **Tokenization & Tagging:**
#    - The input text is tokenized using `word_tokenize()`.
#    - POS tagging is done using `pos_tag()` which outputs a list of tuples.
   
# 4. **Output:**
#    - The tagged tokens are printed.
#    - An optional bar chart of tag frequencies is displayed using matplotlib.

# This script covers all the core steps of the experiment while remaining concise.
