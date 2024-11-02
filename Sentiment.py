from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
# Example sentences
sentences = [
    "I absolutely love this product! It's fantastic and works like a charm.",
    "I'm not too happy with the service; it could be much better.",
    "This is the worst experience I've ever had. Totally disappointed.",
    "It's okay, nothing special but not too bad either.",
]



# Example sentences
sentences = [
    "I absolutely love this product! It's fantastic and works like a charm.",
    "The weather is expected to be sunny tomorrow.",
    "I'm not too happy with the service; it could be much better.",
    "Gravity causes objects to fall toward the Earth.",
]

def spelling(sentence):
    return TextBlob(sentence).correct()

def analysis(sentence):
    blob = TextBlob(sentence)
    sentiment = sia.polarity_scores(sentence)

    return {
        "sentence" : sentence,
        "sentiment" : sentiment["compound"],
        "subjectivity" : blob.subjectivity
    }

# Analyze each sentence
for sentence in sentences:
    data = analysis(sentence)
    print("Sentence: ", data["sentence"])
    print("Sentiment scores:", data["sentiment"])
    print("Subjectivity:", data["subjectivity"])
    


    print("\n")

incorrect_text = "I havv a speling errror."
corrected_text = spelling(incorrect_text)
print(corrected_text)