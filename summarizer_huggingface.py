from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from collections import defaultdict
import spacy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import torch
from scipy.special import softmax
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager 
from datetime import datetime

class ReviewAuthenticityChecker:
    def __init__(self):
        """Initialize models and tools for fake review detection"""
        # BERT model fine-tuned for review authenticity
        self.bert_classifier = pipeline(
            "text-classification",
            model="deepset/bert-base-cased-squad2",
            top_k=None
        )
        
        # Initialize spaCy for linguistic analysis
        self.nlp = spacy.load("en_core_web_sm")
        
        # Linguistic pattern thresholds
        self.thresholds = {
            'min_word_count': 10,
            'max_repeated_phrases': 3,
            'min_unique_words_ratio': 0.7,
            'max_exclamation_marks': 3
        }
    
    def _check_linguistic_patterns(self, review):
        """Analyze linguistic patterns for potential red flags"""
        doc = self.nlp(review)
        
        # Basic text statistics
        word_count = len([token for token in doc if not token.is_punct])
        unique_words = len(set([token.text.lower() for token in doc if not token.is_punct]))
        unique_ratio = unique_words / word_count if word_count > 0 else 0
        
        # Check for repeated phrases
        phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
        phrase_counts = defaultdict(int)
        for phrase in phrases:
            phrase_counts[phrase] += 1
        repeated_phrases = sum(1 for count in phrase_counts.values() if count > 2)
        
        # Count exclamation marks
        exclamation_count = review.count('!')
        
        # Calculate linguistic score
        flags = []
        if word_count < self.thresholds['min_word_count']:
            flags.append('too_short')
        if repeated_phrases > self.thresholds['max_repeated_phrases']:
            flags.append('repetitive_content')
        if unique_ratio < self.thresholds['min_unique_words_ratio']:
            flags.append('limited_vocabulary')
        if exclamation_count > self.thresholds['max_exclamation_marks']:
            flags.append('excessive_punctuation')
        
        return {
            'flags': flags,
            'metrics': {
                'word_count': word_count,
                'unique_ratio': unique_ratio,
                'repeated_phrases': repeated_phrases,
                'exclamation_count': exclamation_count
            }
        }
    
    def _analyze_sentiment_consistency(self, review):
        """Check if sentiment is consistent throughout the review"""
        sentences = [sent.text for sent in self.nlp(review).sents]
        sentiments = [TextBlob(sent).sentiment.polarity for sent in sentences]
        
        # Check for dramatic sentiment shifts
        sentiment_shifts = sum(1 for i in range(len(sentiments)-1) 
                             if abs(sentiments[i] - sentiments[i+1]) > 0.5)
        
        return {
            'sentiment_consistency': sentiment_shifts == 0,
            'sentiment_shifts': sentiment_shifts
        }
    
    def check_review_authenticity(self, review):
        """Comprehensive review authenticity check"""
        # Get BERT-based classification
        bert_scores = self.bert_classifier(review)[0]
        
        # Linguistic pattern analysis
        linguistic_check = self._check_linguistic_patterns(review)
        
        # Sentiment consistency check
        sentiment_analysis = self._analyze_sentiment_consistency(review)
        
        # Calculate overall authenticity score
        # Weight different factors
        weights = {
            'bert_score': 0.5,
            'linguistic_score': 0.3,
            'sentiment_score': 0.2
        }
        
        # Calculate component scores
        bert_authenticity = max([score['score'] for score in bert_scores 
                               if score['label'] == 'AUTHENTIC'], default=0)
        
        linguistic_score = 1.0 - (len(linguistic_check['flags']) / 4)  # Normalize by max possible flags
        sentiment_score = 1.0 if sentiment_analysis['sentiment_consistency'] else \
                         max(0, 1 - (sentiment_analysis['sentiment_shifts'] * 0.2))
        
        # Calculate weighted final score
        authenticity_score = (
            bert_authenticity * weights['bert_score'] +
            linguistic_score * weights['linguistic_score'] +
            sentiment_score * weights['sentiment_score']
        )
        
        return {
            'authenticity_score': round(authenticity_score, 2),
            'is_likely_authentic': authenticity_score >= 0.7,
            'risk_factors': linguistic_check['flags'],
            'metrics': {
                'bert_score': bert_authenticity,
                'linguistic_score': linguistic_score,
                'sentiment_score': sentiment_score,
                **linguistic_check['metrics']
            }
        }
    
class webScraper:
    def webScrape():
        ##a = input('Enter url of product: ')
        a = "https://www.flipkart.com/podge-slim-men-black-jeans/p/itmc3a04001ed4a4?pid=JEAGMNZXYHPDNM2Z&lid=LSTJEAGMNZXYHPDNM2Z4FFFKI&marketplace=FLIPKART&q=jeans&store=clo%2Fvua%2Fk58&srno=s_1_13&otracker=search&otracker1=search&fm=Search&iid=107fbb65-0179-4f54-a8c2-d48077242bf5.JEAGMNZXYHPDNM2Z.SEARCH&ppt=sp&ppn=sp&ssid=wem615s85c0000001731826600737&qH=a0f2589b1ced4dec"
        url = a.replace("/p/", "/product-reviews/")
        url += "&page="

        reviews = []
        #url = "https://www.flipkart.com/xiaomi-14-jade-green-512-gb/product-reviews/itm617acb7cd715d?pid=MOBGYGCMYEJTGEPF&lid=LSTMOBGYGCMYEJTGEPFUFDBXT&marketplace=FLIPKART&store=tyy%2F4io&srno=b_1_1&otracker=nmenu_sub_Electronics_0_Mi&fm=organic&iid=fa38669b-fc0b-4af3-ad7b-9c9d566ed098.MOBGYGCMYEJTGEPF.SEARCH&ppt=None&ppn=None&ssid=qgk59zgwsg0000001725193903709&page="
        i = 1
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        while True:
            driver.get(url+str(i))
            time.sleep(5)
            reviewdiv = driver.find_elements(by = By.CLASS_NAME, value = "EKFha-")
            if(len(reviewdiv) < 1):
                break
            buttons = driver.find_elements(by=By.CLASS_NAME, value="b4x-fr")
            if(len(buttons) > 0): 
                for but in buttons:
                    but.click()
            content = driver.page_source
            soup = BeautifulSoup(content, "html.parser")
            for div in soup.find_all("span", class_="wTYmpv"): 
                div.decompose()
            #print(soup.encode('utf-8'))
            mydivs=soup.find_all('div',class_="ZmyHeo")
            for div in mydivs:
                reviews.append(div.text[1::])
            i = i+1
        return reviews

class EnhancedProductReviewSummarizer:
    def __init__(self):
        # Initialize the summarization model (using BART for better abstractive summaries)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)
        
        # Initialize spaCy for text processing
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize authenticity checker
        self.authenticity_checker = ReviewAuthenticityChecker()
        
        # Initialize vectorizer for keyword extraction
        self.vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2))
    
    def _get_sentiment(self, text):
        """Analyze sentiment of text using TextBlob"""
        return TextBlob(text).sentiment.polarity
    
    def _extract_key_aspects(self, reviews):
        """Extract common aspects mentioned across reviews"""
        if not reviews:
            return []
        import google.generativeai as genai
        genai.configure(api_key='AIzaSyAxapy6Z_v-K5PHaR3HXQHJfArudTlqBQs')
        model = genai.GenerativeModel("gemini-1.5-flash")
        query = "Give key aspects of the product from these reviews in clear concise points of not more that 7 words each"
        for i in reviews:
            query += i+" "
        response = model.generate_content(query)
        #print(response.text)
        return response.text.split("*")
        doc = self.nlp(" ".join(reviews))
        noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
        
        freq_dist = defaultdict(int)
        for phrase in noun_phrases:
            freq_dist[phrase] += 1
            
        generic_terms = {'product', 'item', 'thing', 'review', 'its', 'the', 'that'}
        return [k for k, v in sorted(freq_dist.items(), key=lambda x: x[1], reverse=True) 
                if (k not in generic_terms and len(k) < 2)][:5]
    
    def analyze_reviews(self, reviews, authenticity_threshold=0.4, max_length=150, min_length=50):
        """Analyze reviews with authenticity checking and summarization"""
        if not reviews:
            return {
                'overall_sentiment': 'No reviews',
                'key_aspects': [],
                'integrated_summary': 'No reviews available for analysis.',
                'authenticity_stats': {
                    'total_reviews': 0,
                    'authentic_reviews': 0,
                    'average_authenticity': 0,
                    'removed_reviews': 0
                },
                'review_details': []
            }
        
        # Check authenticity of each review
        review_authenticity = []
        authentic_reviews = []
        
        for review in reviews:
            auth_result = self.authenticity_checker.check_review_authenticity(review)
            review_authenticity.append({
                'review': review,
                'authenticity': auth_result
            })
            
            if auth_result['authenticity_score'] >= authenticity_threshold:
                authentic_reviews.append(review)
        
        # Calculate authenticity statistics
        authenticity_stats = {
            'total_reviews': len(reviews),
            'authentic_reviews': len(authentic_reviews),
            'average_authenticity': np.mean([r['authenticity']['authenticity_score'] 
                                           for r in review_authenticity]),
            'removed_reviews': len(reviews) - len(authentic_reviews)
        }
        
        # Handle case where no authentic reviews are found
        if not authentic_reviews:
            return {
                'overall_sentiment': 'No authentic reviews',
                'key_aspects': [],
                'integrated_summary': 'No authentic reviews found for analysis.',
                'authenticity_stats': authenticity_stats,
                'review_details': review_authenticity
            }
        
        # Calculate overall sentiment
        sentiments = [self._get_sentiment(review) for review in authentic_reviews]
        avg_sentiment = np.mean(sentiments)
        overall_sentiment = 'Positive' if avg_sentiment > 0.1 else 'Negative' if avg_sentiment < -0.1 else 'Mixed'
        
        # Extract key aspects from authentic reviews
        key_aspects = self._extract_key_aspects(authentic_reviews)
        
        try:
            # Prepare reviews for summarization with sentiment context
            formatted_reviews = []
            for review in authentic_reviews:
                sentiment = self._get_sentiment(review)
                if sentiment > 0.1:
                    formatted_reviews.append(f"Positive aspect: {review}")
                elif sentiment < -0.1:
                    formatted_reviews.append(f"Negative aspect: {review}")
                else:
                    formatted_reviews.append(f"Neutral observation: {review}")
            
            # Generate integrated summary from authentic reviews
            combined_reviews = " [SEP] ".join(formatted_reviews)
            integrated_summary = self.summarizer(combined_reviews, 
                                              max_length=max_length,
                                              min_length=min_length,
                                              do_sample=False)[0]['summary_text']
        except Exception as e:
            integrated_summary = f"Error generating summary: {str(e)}"
        
        return {
            'overall_sentiment': overall_sentiment,
            'key_aspects': key_aspects,
            'integrated_summary': integrated_summary,
            'authenticity_stats': authenticity_stats,
            'review_details': review_authenticity
        }

# Example usage
if __name__ == "__main__":
    reviews = webScraper.webScrape()
    print(reviews)
    sample_reviews = [
        "The battery life is amazing, lasting over 10 hours. The screen is bright and crisp. Great value for money!",
        "AMAZING AMAZING AMAZING!!! Best product ever!!!! Must buy!!!! Life changing!!!!!",  # Likely fake
        "Perfect for my needs. Fast processor, runs cool, and the build quality is excellent.",
        "Started having issues after 2 months. The fan is quite loud and customer service wasn't helpful.",
        "Really impressed with the performance. The display is fantastic for creative work.",
        "This product is the best thing ever created in the history of mankind!!!!"  # Likely fake
    ]
    
    # Initialize the enhanced summarizer
    summarizer = EnhancedProductReviewSummarizer()
    
    try:
        # Analyze reviews
        result = summarizer.analyze_reviews(reviews)
        
        # Print results
        print("\nProduct Review Analysis:")
        print(f"Overall Sentiment: {result['overall_sentiment']}")
        
        print("\nAuthenticity Statistics:")
        print(f"Total Reviews: {result['authenticity_stats']['total_reviews']}")
        print(f"Authentic Reviews: {result['authenticity_stats']['authentic_reviews']}")
        print(f"Average Authenticity Score: {result['authenticity_stats']['average_authenticity']:.2f}")
        print(f"Removed Reviews: {result['authenticity_stats']['removed_reviews']}")
        
        print("\nKey Aspects Mentioned:")
        for aspect in result['key_aspects']:
            print(f"- {aspect}")
        
        print("\nIntegrated Summary:")
        print(result['integrated_summary'])
        
                
    except Exception as e:
        print(f"Error during analysis: {str(e)}")


''' print("\nDetailed Review Analysis:")
        for review_detail in result['review_details']:
            print("\nReview:", review_detail['review'][:100], "..." if len(review_detail['review']) > 100 else "")
            print(f"Authenticity Score: {review_detail['authenticity']['authenticity_score']}")
            if review_detail['authenticity']['risk_factors']:
                print("Risk Factors:", review_detail['authenticity']['risk_factors'])'''