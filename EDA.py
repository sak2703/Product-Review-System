import pandas as pd 
import numpy as np 
import nltk 
from tqdm import tqdm  
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from nltk.sentiment import SentimentIntensityAnalyzer
maxsentimentscore=0;
product=0;
def classify_sentiment(review):
    
    # Load positive and negative words
    positive_words = set(load_data('positive_words.txt'))
    negative_words = set(load_data('negative_words.txt'))
    
    # Tokenize the review
    words = review.lower().split()
   

    # Initialize sentiment score
    sentiment_score = 0

    # Iterate through the words in the review
    idx = 0
    while idx < len(words):
        word = words[idx]
        if word in positive_words:
            sentiment_score += 1
        elif word in negative_words:
            sentiment_score -= 1
        # Check for edge cases where a positive word is followed by a negative word
        elif word == "pretty" and idx < len(words) - 1 and words[idx + 1] in negative_words:
            sentiment_score -= 1
        elif word == "bad" and idx > 0 and words[idx - 1] in positive_words:
            sentiment_score -= 1
        idx += 1
        return sentiment_score

    

    # Classify based on sentiment score
    
def load_data(filename):
    reviews = []
    with open(filename, 'r') as file:
        for line in file:
            reviews.append(line.strip())
    return reviews

    # Load reviews from file
reviews = load_data('reviews.txt')


    # Classify each review and print the result




def get_vader_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    vader_result = sia.polarity_scores(text)
    return vader_result

def get_roberta_sentiment(text):
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict


# Define weights based on RMSE and MAE performance metrics
weight_vader = 0.6
weight_roberta = 0.3 
weight_classify = 0.1  

# Function to calculate overall sentiment score
def calculate_overall_sentiment(vader_score, roberta_score, classify_score):
    overall_score = (
        weight_vader * vader_score['compound'] + 
        weight_roberta * (roberta_score['roberta_neg'] + roberta_score['roberta_pos']) / 2 + 
        weight_classify * classify_score
    )
    return overall_score

# Example usage
for idx, review in enumerate(reviews, start=1):
    vader_score = get_vader_sentiment(review)
    roberta_score = get_roberta_sentiment(review)
    classify_score = classify_sentiment(review)
    overall_sentiment = calculate_overall_sentiment(vader_score, roberta_score, classify_score)
    print(f"Review {idx}: Overall Sentiment Score: {overall_sentiment}")
    if(maxsentimentscore<overall_sentiment):
        product=idx
        maxsentimentscore=overall_sentiment

print(f"Recommended Product is:{product}")
    

        
        


