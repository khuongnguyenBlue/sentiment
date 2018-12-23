import pickle
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from evaluation import get_metrics
from load_data import load_dataset


def save_model(filename, classifier):
    with open(filename, 'wb') as f:
        pickle.dump(classifier, f)


print("Loading data...")
X, Y = load_dataset()

print("Training model")
t0 = time()
transformer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.5)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = transformer.fit_transform(X_train)
X_test = transformer.transform(X_test)

model = MultinomialNB()

clf = model.fit(X_train, y_train)
train_time = time() - t0
print("Finished")
print("\t- train time: %0.3fs" % train_time)

t0 = time()
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
test_time = time() - t0
print("\t- test time: %0.3fs" % test_time)

get_metrics(y_test, y_pred)
save_model("model/model.pkl", clf)
