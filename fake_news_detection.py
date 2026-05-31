
import pandas as pd

df = pd.read_csv("evaluation.csv", on_bad_lines='skip', delimiter=';')
df = df.dropna()

df.describe()

X = df["text"]
Y = df["label"]

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words="english" , max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

print(X_vectorized)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_vectorized, Y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, y_pred))

def predict_news(text):
    vec = vectorizer.transform([text])
    return model.predict(vec)

print(predict_news("Government announces new education reform"))
print(predict_news("Aliens landed in Algeria yesterday and took control"))