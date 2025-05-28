import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

data = {
    'text': [
        "I love this product, it's amazing!",
        "This is the worst experience I have ever had.",
        "It is okay, not great but not bad either.",
        "Absolutely fantastic service!",
        "I am not happy with the quality.",
        "Nothing special, just average.",
        "Best purchase I've made this year.",
        "Terrible! I want a refund.",
        "Meh, it's fine.",
        "I feel great about this.",
        "It was a horrible day.",
        "The movie was dreadful and boring.",
        "I hate this item.",
        "I'm satisfied with the purchase.",
        "The food was awful.",
        "The experience was not good.",
        "Pretty decent and enjoyable.",
        "It could have been better.",
        "I'm disappointed.",
        "Everything was perfect."
    ],
    'sentiment': [
        'positive',
        'negative',
        'neutral',
        'positive',
        'negative',
        'neutral',
        'positive',
        'negative',
        'neutral',
        'positive',
        'negative',
        'negative',
        'negative',
        'positive',
        'negative',
        'negative',
        'positive',
        'neutral',
        'negative',
        'positive'
    ]
}

df = pd.DataFrame(data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

def predict_sentiment_with_probs(text):
    text_tfidf = vectorizer.transform([text])
    probs = model.predict_proba(text_tfidf)[0]  # probabilities for each class
    classes = model.classes_  # ['negative', 'neutral', 'positive']
    # Create dict of class:prob
    prob_dict = {cls: round(prob, 4) for cls, prob in zip(classes, probs)}
    predicted_class = model.predict(text_tfidf)[0]
    return predicted_class, prob_dict

test_sentences = [
    "I really enjoyed this movie!",
    "It was a horrible day.",
    "The product is okay, nothing special.",
    "The food was awful and disgusting.",
    "Absolutely loved the concert!",
    "I'm very disappointed with the service."
]

for sentence in test_sentences:
    pred, probas = predict_sentiment_with_probs(sentence)
    print(f"Text: {sentence}")
    print(f" Predicted Sentiment: {pred}")