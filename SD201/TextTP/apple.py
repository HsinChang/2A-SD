import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics

#load the training directory and process the data
trainingFiles = load_files('Training')
apple_train, apple_test, y_train, y_test = train_test_split(
    trainingFiles.data, trainingFiles.target, test_size=0.33)
count_vect = CountVectorizer(analyzer='word', binary= True, max_features=200,
                             #stop_words = frozenset(["is","are","and","or","a","an","the","and","of","for","this","that","to","have","has","not","as","at","by"]))
                             stop_words='english')
X_train_counts = count_vect.fit_transform(apple_train)
training = X_train_counts.toarray()
X_test_counts = count_vect.fit_transform(apple_test)
testing = X_test_counts.toarray()

#the classification and classification report
neigh = KNeighborsClassifier(n_neighbors = 8)
neigh.fit(training, y_train)
y_predicted = neigh.predict(testing)
print(metrics.classification_report(y_test, y_predicted,
                                    target_names=trainingFiles.target_names))

#the real test

