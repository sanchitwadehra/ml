import nltk
# Uncomment the next line if you haven't already downloaded the 'punkt' tokenizer
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import matplotlib.pyplot as plt

def generate_ngrams(text, n):
    """
    Generate n-grams from the provided text.
    """
    tokens = word_tokenize(text.lower())
    return list(ngrams(tokens, n))

if __name__ == "__main__":
    # Sample input text for the experiment
    text = (
        "Natural language processing is essential for text prediction, language modeling, "
        "and understanding patterns in data. This experiment demonstrates the generation "
        "and analysis of unigrams, bigrams, and trigrams."
    )
    
    # Tokenize text and generate n-grams
    tokens = word_tokenize(text.lower())
    unigrams = tokens
    bigrams = generate_ngrams(text, 2)
    trigrams = generate_ngrams(text, 3)
    
    # Frequency analysis of n-grams using Counter
    unigram_freq = Counter(unigrams)
    bigram_freq = Counter(bigrams)
    trigram_freq = Counter(trigrams)
    
    # Display the top 5 most frequent n-grams
    print("Top 5 Unigrams:", unigram_freq.most_common(5))
    print("Top 5 Bigrams:", bigram_freq.most_common(5))
    print("Top 5 Trigrams:", trigram_freq.most_common(5))
    
    # Optional: Visualization of the top 5 bigrams
    top_bigrams = bigram_freq.most_common(5)
    bigram_labels = [" ".join(bigram) for bigram, freq in top_bigrams]
    frequencies = [freq for bigram, freq in top_bigrams]
    
    plt.figure(figsize=(8, 5))
    plt.bar(bigram_labels, frequencies, color='skyblue')
    plt.title("Top 5 Most Frequent Bigrams")
    plt.xlabel("Bigrams")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    
    
# Below is a concise Python script (e.g., in your `S-6 NLP/Exp-4.py` file) that covers the experiment by tokenizing text, generating unigrams, bigrams, and trigrams, performing frequency analysis using `Counter`, and visualizing the top 5 bigrams with `matplotlib`.

# ```python:S-6 NLP/Exp-4.py
# import nltk
# # Uncomment the next line if you haven't already downloaded the 'punkt' tokenizer
# # nltk.download('punkt')
# from nltk.tokenize import word_tokenize
# from nltk.util import ngrams
# from collections import Counter
# import matplotlib.pyplot as plt

# def generate_ngrams(text, n):
#     """
#     Generate n-grams from the provided text.
#     """
#     tokens = word_tokenize(text.lower())
#     return list(ngrams(tokens, n))

# if __name__ == "__main__":
#     # Sample input text for the experiment
#     text = (
#         "Natural language processing is essential for text prediction, language modeling, "
#         "and understanding patterns in data. This experiment demonstrates the generation "
#         "and analysis of unigrams, bigrams, and trigrams."
#     )
    
#     # Tokenize text and generate n-grams
#     tokens = word_tokenize(text.lower())
#     unigrams = tokens
#     bigrams = generate_ngrams(text, 2)
#     trigrams = generate_ngrams(text, 3)
    
#     # Frequency analysis of n-grams using Counter
#     unigram_freq = Counter(unigrams)
#     bigram_freq = Counter(bigrams)
#     trigram_freq = Counter(trigrams)
    
#     # Display the top 5 most frequent n-grams
#     print("Top 5 Unigrams:", unigram_freq.most_common(5))
#     print("Top 5 Bigrams:", bigram_freq.most_common(5))
#     print("Top 5 Trigrams:", trigram_freq.most_common(5))
    
#     # Optional: Visualization of the top 5 bigrams
#     top_bigrams = bigram_freq.most_common(5)
#     bigram_labels = [" ".join(bigram) for bigram, freq in top_bigrams]
#     frequencies = [freq for bigram, freq in top_bigrams]
    
#     plt.figure(figsize=(8, 5))
#     plt.bar(bigram_labels, frequencies, color='skyblue')
#     plt.title("Top 5 Most Frequent Bigrams")
#     plt.xlabel("Bigrams")
#     plt.ylabel("Frequency")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()
# ```

# ### Explanation

# - **Tokenization:** The script uses `nltk.word_tokenize` to convert text into tokens.
# - **N-gram Generation:** The `generate_ngrams()` function generates n-grams (both bi- and trigrams in this case) from the text.
# - **Frequency Analysis:** `collections.Counter` is used to count the occurrences of each n-gram.
# - **Visualization:** A bar graph is created using `matplotlib` to visualize the top 5 most frequent bigrams.

# This short code snippet covers the main steps of your experiment, from text preprocessing to n-gram analysis and visualization.
