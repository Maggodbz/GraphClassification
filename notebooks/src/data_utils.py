import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Define a list of common English stopwords
stop_words = set([
    "ourselves", "hers", "between", "yourself", "but",
    "again", "there", "about", "once", "during", "out",
    "very", "having", "with", "they", "own", "an", "be",
    "some", "for", "do", "its", "yours", "such", "into",
    "of", "most", "itself", "other", "off", "is", "s", "am",
    "or", "who", "as", "from", "him", "each", "the", "themselves",
    "until", "below", "are", "we", "these", "your", "his", "through",
    "don", "nor", "me", "were", "her", "more", "himself", "this",
    "down", "should", "our", "their", "while", "above", "both", "up",
    "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before",
    "them", "same", "and", "been", "have", "in", "will", "on", "does",
    "yourselves", "then", "that", "because", "what", "over", "why", "so",
    "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just",
    "where", "too", "only", "myself", "which", "those", "i", "after", "few",
    "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing",
    "it", "how", "further", "was", "here", "than"
])

# Function to preprocess text


def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize and remove stop words
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Return the processed text
    return ' '.join(words)
