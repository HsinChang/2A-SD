import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from sklearn.metrics import accuracy_score

trainingFiles = load_files('TechFruits', shuffle=True)
data_train = trainingFiles.data
data_class = trainingFiles.target
count_vect = TfidfVectorizer(analyzer='word')

train_counts = count_vect.fit_transform(data_train)
training = train_counts.toarray()

clf = MultinomialNB()
# clf.fit(training, data_class)

k = 3
training_list = []
class_list = []
spilt_size = int(training.shape[0] / k)
#split the data into two lists
for i in range(0, k):
    training_list.append(training[i * spilt_size:(i + 1) * spilt_size])
    class_list.append(data_class[i * spilt_size:(i + 1) * spilt_size])
    #np.append(training[i * spilt_size:(i + 1) * spilt_size], axis =0)
    #locals()['class_list'+str(i)] = np.empty(0)
    #np.append(data_class[i * spilt_size:(i + 1) * spilt_size], axis = 0)
sum_accuracy = 0
for i in range(0, k):
    temp_training = []
    temp_class = []
    temp_test = []
    temp_class_test = []
    for j in range(0, k):
        if (j != i):
            temp_training.extend(training_list[j])
            temp_class.extend(class_list[j])
        else:
            temp_test.extend(training_list[j])
            temp_class_test.extend(class_list[j])
    temp_training = np.concatenate(temp_training, axis= 0)
    temp_training = np.reshape(temp_training, (spilt_size * (k -1),training.shape[1]))
    temp_test = np.concatenate(temp_test, axis = 0)
    temp_test = np.reshape(temp_test, (spilt_size,training.shape[1]))
    temp_class = np.array(temp_class)
    temp_class_test = np.array(temp_class_test)
    clf.fit(temp_training, temp_class)
    temp_predicted = clf.predict(temp_test)
    sum_accuracy += accuracy_score(temp_class_test, temp_predicted)

print("The accuracy score is " + str(sum_accuracy/k))
