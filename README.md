#InsightFuse- Product Review System

A comprehensive product review analysis tool that leverages cutting-edge Natural Language Processing (NLP) techniques to detect fake reviews, analyze sentiment, and summarize authentic feedback from e-commerce platforms. The project uses state-of-the-art machine learning models like BERT for classification and BART for summarization.

## Features

1. **Fake Review Detection:**
   - Uses BERT (fine-tuned for classification tasks) to identify potentially inauthentic reviews based on linguistic patterns and content analysis.
   - Custom thresholds for detecting repetitive phrases, limited vocabulary, and excessive punctuation.

2. **Sentiment Analysis:**
   - Employs TextBlob to measure polarity and assess sentiment consistency across reviews.
   - Detects sentiment shifts and flags overly dramatic changes in tone.

3. **Summarization:**
   - Utilizes the BART model for abstractive summarization to generate concise, meaningful summaries of authentic reviews.
   - Extracts key aspects of the product through NLP techniques like noun phrase extraction and frequency analysis.

4. **Web Scraping:**
   - Implements Selenium and BeautifulSoup to scrape product reviews directly from e-commerce websites like Flipkart.
   - Handles dynamic content loading and ensures the extraction of all relevant review text.

5. **Comprehensive Review Analysis:**
   - Filters reviews based on authenticity score and provides integrated summaries, sentiment insights, and key aspects.

---

## Tech Stack

### Backend
- **Python**
- **Hugging Face Transformers**: BERT (deepset/bert-base-cased-squad2) and BART (facebook/bart-large-cnn)
- **TextBlob**: Sentiment analysis
- **spaCy**: Linguistic analysis
- **Selenium**: Web scraping
- **BeautifulSoup**: HTML parsing
- **CountVectorizer**: Keyword extraction

### Tools and Libraries
- **Numpy**
- **Scikit-learn**
- **Selenium WebDriver Manager**
- **Google Generative AI** (for key aspect generation)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sak2703/Product-Review-System.git
   cd Product-Review-System
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install additional NLP models:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. Set up Selenium WebDriver (Chrome):
   - Ensure **Google Chrome** is installed.
   - The script uses WebDriver Manager to automatically set up the driver.

---

## Usage

### Web Scraping
The `webScraper` class scrapes product reviews from Flipkart.

```python
from summarizer_huggingface import webScraper

# Scrape reviews
reviews = webScraper.webScrape()
```

### Review Analysis and Summarization
The `EnhancedProductReviewSummarizer` class performs the review analysis and generates insights.

```python
from summarizer_huggingface import EnhancedProductReviewSummarizer

# Initialize the summarizer
summarizer = EnhancedProductReviewSummarizer()

# Analyze reviews
results = summarizer.analyze_reviews(reviews)

# Display results
print("Overall Sentiment:", results['overall_sentiment'])
print("Key Aspects:", results['key_aspects'])
print("Integrated Summary:", results['integrated_summary'])
```

---

## Output
The tool provides:
- Overall sentiment of the reviews (Positive, Negative, Mixed).
- Key aspects extracted from authentic reviews.
- Integrated summary of authentic reviews.
- Authenticity statistics (e.g., total reviews, authentic reviews, average authenticity score).
- Detailed authenticity and risk factor analysis for each review.

---

## Example
### Input
A Flipkart product URL for web scraping.

### Output
```plaintext
Overall Sentiment: Positive
Key Aspects:
- Great battery life
- Bright and crisp screen
- Excellent build quality
Integrated Summary:
"The product has amazing battery life, a vibrant screen, and excellent build quality, making it a great value for money."
Authenticity Statistics:
Total Reviews: 100
Authentic Reviews: 85
Average Authenticity Score: 0.78
Removed Reviews: 15
```

---

## Future Enhancements
- Support for other e-commerce platforms like Amazon.
- Integration of advanced models for fake review detection.
- Real-time review monitoring and analysis dashboard.
- Multilingual review analysis support.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests to improve the project.

---

## Contact
For queries or feedback, reach out to [Sak2703](https://github.com/sak2703).

