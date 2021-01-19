import time
import inline as inline
import matplotlib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
import graphviz
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree, preprocessing
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.neural_network import MLPClassifier
import seaborn as sns
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import accuracy_score

sns.set()
df = pd.read_csv('student-mat.csv')
sp = pd.read_csv('student-por.csv')
df = df.append(sp, ignore_index=True)
my_labels = 'Male','Female'
print(df.groupby('activities').count()['school'])
sex = df.groupby('activities').count()['school']
labels = "Yes", "No"
plt.pie(sex, labels= labels, autopct='%1.1f%%')
plt.title('Extra-curricular activities')
plt.axis('equal')
plt.savefig('activities.png')
plt.show()
plt.close()
print(df)
print(df.isnull().any())
print(df.dtypes)
print(df)
le = preprocessing.LabelEncoder()
ds = df.apply(le.fit_transform)
df['school'] = ds['school']
df['sex'] = ds['sex']
df['address'] = ds['address']
df['famsize'] = ds['famsize']
df['Pstatus'] = ds['Pstatus']
df['Mjob'] = ds['Mjob']
df['Fjob'] = ds['Fjob']
df['guardian'] = ds['guardian']
df['schoolsup'] = ds['schoolsup']
df['famsup'] = ds['famsup']
df['paid'] = ds['paid']
df['activities'] = ds['activities']
df['nursery'] = ds['nursery']
df['higher'] = ds['higher']
df['internet'] = ds['internet']
df['romantic'] = ds['romantic']
df['reason'] = ds['reason']
print(df)
columns = list(df.columns)
classes_list = ['Pstatus', 'school', 'schoolsup', 'famsup', 'activities', 'nursery', 'higher', 'romantic']
# classes_list = ['Pstatus']
score_tree = []
score_naive = []
score_knn3 = []
score_knn5 = []
score_knn11 = []
score_neural = []
score_forest = []
score_svc = []
time_tree = []
time_naive = []
time_knn3 = []
time_knn5 = []
time_knn11 = []
time_neural = []
time_forest = []
time_svc = []
for column in classes_list:
    columns.pop(columns.index(column))
    all_inputs = df[columns]
    all_classes = df[column].values
    (train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.3,
                                                                                random_state=1)
    print("Class:", column)
    # DecisionTree Clasification

    dtc = DecisionTreeClassifier()
    start = time.time()
    dtc.fit(train_inputs, train_classes)
    predictions = dtc.predict(test_inputs)
    print("DecisionTree:", dtc.score(test_inputs, test_classes))
    end = time.time()
    time_tree.append(end - start)
    predicted = cross_val_predict(dtc, all_inputs, all_classes)
    class_labels = list(set(predicted))
    model_cm = confusion_matrix(y_true=all_classes, y_pred=predicted, labels=class_labels)
    print(model_cm)
    score_tree.append(dtc.score(test_inputs, test_classes))

    gnb = GaussianNB()
    start = time.time()
    gnb.fit(train_inputs, train_classes)
    print("Naive Bayes:", gnb.score(test_inputs, test_classes))
    end = time.time()
    time_naive.append(end - start)
    predicted = cross_val_predict(gnb, all_inputs, all_classes)
    class_labels = list(set(predicted))
    model_cm = confusion_matrix(y_true=all_classes, y_pred=predicted, labels=class_labels)
    print(model_cm)
    score_naive.append(gnb.score(test_inputs, test_classes))

    mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=2000)
    start = time.time()
    mlp.fit(train_inputs, train_classes)
    print("Neural:", mlp.score(test_inputs, test_classes))
    end = time.time()
    time_neural.append(end - start)
    predicted = cross_val_predict(mlp, all_inputs, all_classes)
    class_labels = list(set(predicted))
    model_cm = confusion_matrix(y_true=all_classes, y_pred=predicted, labels=class_labels)
    print(model_cm)
    score_neural.append(mlp.score(test_inputs, test_classes))

    knn3 = KNeighborsClassifier(n_neighbors=3)
    start = time.time()
    knn3.fit(train_inputs, train_classes)
    print("Knn3:", knn3.score(test_inputs, test_classes))
    end = time.time()
    time_knn3.append(end - start)
    predicted = cross_val_predict(knn3, all_inputs, all_classes)
    class_labels = list(set(predicted))
    model_cm = confusion_matrix(y_true=all_classes, y_pred=predicted, labels=class_labels)
    print(model_cm)
    score_knn3.append(knn3.score(test_inputs, test_classes))

    knn5 = KNeighborsClassifier(n_neighbors=5)
    start = time.time()
    knn5.fit(train_inputs, train_classes)
    print("Knn5:", knn5.score(test_inputs, test_classes))
    end = time.time()
    time_knn5.append(end - start)
    predicted = cross_val_predict(knn5, all_inputs, all_classes)
    class_labels = list(set(predicted))
    model_cm = confusion_matrix(y_true=all_classes, y_pred=predicted, labels=class_labels)
    print(model_cm)
    score_knn5.append(knn5.score(test_inputs, test_classes))

    knn11 = KNeighborsClassifier(n_neighbors=11)
    start = time.time()
    knn11.fit(train_inputs, train_classes)
    print("Knn11:", knn11.score(test_inputs, test_classes))
    end = time.time()
    time_knn11.append(end - start)
    predicted = cross_val_predict(knn11, all_inputs, all_classes)
    class_labels = list(set(predicted))
    model_cm = confusion_matrix(y_true=all_classes, y_pred=predicted, labels=class_labels)
    print(model_cm)
    score_knn11.append(knn11.score(test_inputs, test_classes))

    svc = SVC()
    start = time.time()
    svc.fit(train_inputs, train_classes)
    print("Support Vector:", svc.score(test_inputs, test_classes))
    end = time.time()
    time_svc.append(end - start)
    predicted = cross_val_predict(svc, all_inputs, all_classes)
    class_labels = list(set(predicted))
    model_cm = confusion_matrix(y_true=all_classes, y_pred=predicted, labels=class_labels)
    print(model_cm)
    score_svc.append(svc.score(test_inputs, test_classes))

    rfc = RandomForestClassifier()
    start = time.time()
    rfc.fit(train_inputs, train_classes)
    print("Forest:", rfc.score(test_inputs, test_classes))
    end = time.time()
    time_forest.append(end - start)
    predicted = cross_val_predict(rfc, all_inputs, all_classes)
    class_labels = list(set(predicted))
    model_cm = confusion_matrix(y_true=all_classes, y_pred=predicted, labels=class_labels)
    print(model_cm)
    score_forest.append(rfc.score(test_inputs, test_classes))

    columns = list(df.columns)

print("Decision Tree:", np.mean(score_tree))
print("Time:", np.mean(time_tree))
print("Naive Bayes:", np.mean(score_naive))
print("Time:", np.mean(time_naive))
print("Neural:", np.mean(score_neural))
print("Time:", np.mean(time_neural))
print("Knn3:", np.mean(score_knn3))
print("Time:", np.mean(time_knn3))
print("Knn5:", np.mean(score_knn5))
print("Time:", np.mean(time_knn5))
print("Knn11:", np.mean(score_knn11))
print("Time:", np.mean(time_knn11))
print("Support Vector:", np.mean(score_svc))
print("Time:", np.mean(time_svc))
print("Forest:", np.mean(score_forest))
print("Time:", np.mean(time_forest))

labels = ['DTC', 'NB', 'MLP',
          'Knn3', 'Knn5', 'Knn11', 'SVC', 'RFC']
scores = [np.mean(score_tree), np.mean(score_naive), np.mean(score_neural), np.mean(score_knn3),
          np.mean(score_knn5), np.mean(score_knn11), np.mean(score_svc), np.mean(score_forest)]
# scores = list(map(lambda x: round(x * 100, 1), scores))
times = [np.mean(time_tree), np.mean(time_naive), np.mean(time_neural), np.mean(time_knn3),
         np.mean(time_knn5), np.mean(time_knn11), np.mean(time_svc), np.mean(time_forest)]

plot_space = max(scores) - min(scores)
plt.ylim([min(scores) - plot_space, max(scores) + plot_space])
plt.bar(labels, scores)
for i, v in enumerate(scores):
    plt.text(i - .3, v + 3, str(v))
plt.xlabel('Klasyfikatory')
plt.ylabel('Dokładność')
plt.title('Spożycie alkoholu przez studentów - Średnie Dokładności klasyfikatorów')
plt.savefig('accuracy_Normalized.png')
plt.show()

plot_space = max(times) - min(times)
plt.bar(labels, times)
plt.xlabel('Klasyfikatory')
plt.ylabel('Czas (sekundy)')
plt.title('Spożycie alkoholu przez studentów - Średnie Czasy Działania klasyfikatorów')
plt.savefig('timing_Normalized.png')
plt.show()
