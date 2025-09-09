
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# ===============================
# Task 1: Movie Recommendation (Content-Based)
# ===============================
def task1():
    print("\n--- Task 1: Movie Recommendation (Content-Based) ---")
    data = pd.DataFrame({
        "title": ["Inception", "Interstellar", "The Dark Knight", "Avengers", "Iron Man"],
        "genre": ["Sci-Fi Thriller", "Sci-Fi Drama", "Action Crime", "Action Superhero", "Action Superhero"]
    })
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(data["genre"])
    sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    idx = data[data["title"] == "Inception"].index[0]
    scores = list(enumerate(sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    print("Movies similar to Inception:")
    for i in scores[1:3]:
        print("-", data.iloc[i[0]]["title"])

# ===============================
# Task 2: Movie Recommendation (Collaborative Filtering - Simple)
# ===============================
def task2():
    print("\n--- Task 2: Movie Recommendation (Collaborative Filtering) ---")
    ratings = pd.DataFrame({
        "userId": [1, 1, 2, 2, 3, 3],
        "movie": ["Inception", "Avengers", "Inception", "Iron Man", "Avengers", "The Dark Knight"],
        "rating": [5, 4, 4, 5, 5, 4]
    })
    user_ratings = ratings.pivot_table(index="userId", columns="movie", values="rating")
    corr = user_ratings.corr(min_periods=1)
    print("Movies similar to Inception:")
    print(corr["Inception"].dropna().sort_values(ascending=False))

# ===============================
# Task 3: Spam Email Classifier
# ===============================
def task3():
    print("\n--- Task 3: Spam Email Classifier ---")
    data = pd.DataFrame({
        "text": ["Win money now", "Hello friend how are you", "Free lottery ticket", "Let's meet tomorrow"],
        "label": ["spam", "ham", "spam", "ham"]
    })
    X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.5)
    cv = CountVectorizer()
    X_train_cv = cv.fit_transform(X_train)
    X_test_cv = cv.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train_cv, y_train)
    preds = model.predict(X_test_cv)
    print("Accuracy:", accuracy_score(y_test, preds))

# ===============================
# Task 4: Sentiment Analysis (IMDB - Sample)
# ===============================
def task4():
    print("\n--- Task 4: Sentiment Analysis ---")
    data = pd.DataFrame({
        "review": ["I loved the movie", "Worst movie ever", "Amazing storyline", "Terrible acting"],
        "sentiment": ["positive", "negative", "positive", "negative"]
    })
    X_train, X_test, y_train, y_test = train_test_split(data["review"], data["sentiment"], test_size=0.5)
    cv = CountVectorizer()
    X_train_cv = cv.fit_transform(X_train)
    X_test_cv = cv.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train_cv, y_train)
    preds = model.predict(X_test_cv)
    print("Accuracy:", accuracy_score(y_test, preds))

# ===============================
# Task 5: Stock Price Prediction (Simple Linear Regression)
# ===============================
def task5():
    print("\n--- Task 5: Stock Price Prediction ---")
    days = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    price = np.array([100, 102, 104, 106, 108, 110])
    model = LinearRegression()
    model.fit(days, price)
    future = np.array([[7]])
    print("Predicted price on day 7:", model.predict(future)[0])

# ===============================
# Task 6: Digit Recognition (MNIST + Neural Network)
# ===============================
def task6():
    print("\n--- Task 6: Digit Recognition ---")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train[:5000], y_train[:5000], epochs=2, verbose=0)
    loss, acc = model.evaluate(X_test[:1000], y_test[:1000], verbose=0)
    print("Test Accuracy:", acc)

# ===============================
# Main Menu
# ===============================
if __name__ == "__main__":
    while True:
        print("\nChoose Task:")
        print("1 - Movie Recommendation (Content)")
        print("2 - Movie Recommendation (Collaborative)")
        print("3 - Spam Email Classifier")
        print("4 - Sentiment Analysis")
        print("5 - Stock Price Prediction")
        print("6 - Digit Recognition (MNIST)")
        print("0 - Exit")
        choice = input("Enter choice: ")
        
        if choice == "1": task1()
        elif choice == "2": task2()
        elif choice == "3": task3()
        elif choice == "4": task4()
        elif choice == "5": task5()
        elif choice == "6": task6()
        elif choice == "0":
            print("Exiting..."); break
        else:
            print("Invalid choice!")
