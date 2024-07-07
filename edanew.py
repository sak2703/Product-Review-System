import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

def classify_sentiment(review):
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Tokenize the review and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(review.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    
    # Initialize sentiment score
    sentiment_score = 0
    
    # Iterate through words and calculate sentiment score
    for word in words:
        sentiment_score += sia.polarity_scores(word)['compound']
    
    # Classify sentiment based on score
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"

# Load reviews from file
reviews = load_data('reviews.txt')

# Classify each review and print the result
positive_count = 0
negative_count = 0
for idx, review in enumerate(reviews, start=1):
    sentiment = classify_sentiment(review)
    print(f"Review {idx}: {sentiment}")
    if sentiment == "Positive":
        positive_count += 1
    elif sentiment == "Negative":
        negative_count += 1

# Calculate overall sentiment score
overall_score = positive_count - negative_count
print(f"Overall product score: {overall_score}")
