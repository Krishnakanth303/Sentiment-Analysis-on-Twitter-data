import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download necessary NLTK data (only needed once)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

def analyze_sentiment(tweet):
    # Initialize the VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    
    # Get sentiment scores
    scores = sid.polarity_scores(tweet)
    
    # Determine sentiment category
    if scores['compound'] >= 0.05:
        sentiment = "Positive"
    elif scores['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    # Return detailed results
    result = {
        'text': tweet,
        'sentiment': sentiment,
        'scores': {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound']
        }
    }
    
    return result

# Example usage
if __name__ == "__main__":


    sample_tweets = [
        "I love this new feature! It's amazing and works perfectly.",
        "This is the worst service I've ever experienced. Terrible customer support.",
        "The weather is cloudy today. Might rain later.",
        "Just upgraded my account. Hope it's worth it."
    ]
    
    print("Sample Sentiment Analysis Results:")
    print("---------------------------------")
    
    for tweet in sample_tweets:
        result = analyze_sentiment(tweet)
        print(f"Tweet: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Scores: Positive={result['scores']['positive']:.3f}, "
              f"Negative={result['scores']['negative']:.3f}, "
              f"Neutral={result['scores']['neutral']:.3f}, "
              f"Compound={result['scores']['compound']:.3f}")
        print("---------------------------------")