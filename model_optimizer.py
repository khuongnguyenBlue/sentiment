from time import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from load_data import load_dataset


def grid_search(pipeline):
    score = 'f1_macro'
    gsearch = GridSearchCV(pipeline, parameters, cv=5, scoring=score, n_jobs=-1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)
    t0 = time()
    gsearch.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()
    print("Best dev score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    print("Best test score: %0.3f" % gsearch.score(X_test, y_test))


print("Loading from dataset")
X, Y = load_dataset()
print('y[0]: ' + str(Y.count(0)))
print('y[1]: ' + str(Y.count(1)))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

pipeline_tfidf = Pipeline([
    ("vect", TfidfVectorizer()),
    ("clf", MultinomialNB()),
])
pipeline_count = Pipeline([
    ("vect", CountVectorizer()),
    ("clf", MultinomialNB()),
])
parameters = {
    'vect__max_df': (0.5, 0.6, 0.7, 0.8),
    'vect__ngram_range': ((1, 2), (1, 3)),
}

print("For tfidf_vectorizer:")
grid_search(pipeline_tfidf)

print("For count_vectorizer:")
grid_search(pipeline_count)


