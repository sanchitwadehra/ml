import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

def remove_stopwords(text, language):
    # Tokenize the input text into words
    words = word_tokenize(text)
    
    # Get the list of stopwords for the specified language
    stop_words = set(stopwords.words(language))
    
    # Remove stopwords from the list of words
    filtered_words = [word for word in words if word.lower() not in stop_words]
    
    # Join the filtered words to form the cleaned text
    cleaned_text = ' '.join(filtered_words)
    return cleaned_text

def main():
    # Example texts in different languages
    english_text = "This is an example sentence in English."
    spanish_text = "Este es un ejemplo de frase en español."
    french_text = "Ceci est un exemple de phrase en français."
    german_text = "Dies ist ein Beispiel für einen Satz in Deutsch."
    italian_text = "Questo è un esempio di frase in italiano."

    # Process texts in different languages
    english_result = remove_stopwords(english_text, 'english')
    spanish_result = remove_stopwords(spanish_text, 'spanish')
    french_result = remove_stopwords(french_text, 'french')
    german_result = remove_stopwords(german_text, 'german')
    italian_result = remove_stopwords(italian_text, 'italian')

    # Print results
    print("\nOriginal and processed texts:")
    print("\nEnglish:")
    print("Original:", english_text)
    print("Processed:", english_result)
    
    print("\nSpanish:")
    print("Original:", spanish_text)
    print("Processed:", spanish_result)
    
    print("\nFrench:")
    print("Original:", french_text)
    print("Processed:", french_result)
    
    print("\nGerman:")
    print("Original:", german_text)
    print("Processed:", german_result)
    
    print("\nItalian:")
    print("Original:", italian_text)
    print("Processed:", italian_result)

if __name__ == "__main__":
    main()
