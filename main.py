import inline as inline
import matplotlib
#%matplotlib inline
import os

from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB

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
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import seaborn as sns
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import accuracy_score


sns.set()
df = pd.read_csv('student-mat.csv')
print(df.isnull().any())
print(df.dtypes)
print(df)
le = preprocessing.LabelEncoder()
ds = df.apply(le.fit_transform)
df['school'] = ds['school']
#df['sex'] = ds['sex']
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
all_inputs = df[['school', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason',
                 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities',
                 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc','health',
                 'absences', 'G1', 'G2', 'G3']].values
all_classes = df['sex'].values

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.3, random_state=1)

print(train_inputs)
print(train_classes)
features = ('school', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason',
                 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities',
                 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc','health',
                 'absences', 'G1', 'G2', 'G3')
classes = ('sex')

dtc = DecisionTreeClassifier()
dtc.fit(train_inputs, train_classes)
print(dtc.score(test_inputs, test_classes))


X, y = test_inputs, test_classes
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

print(tree.plot_tree(clf))

#dot_data = tree.export_graphviz(clf, out_file=None)
#graph = graphviz.Source(dot_data)
iris = load_iris()
#print(iris)
#print(df)
dot_data = tree.export_graphviz(clf, out_file=None,
                      feature_names=features,
                      class_names=test_classes,
                      filled=True, rounded=True,
                      special_characters=True)
#dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("TreeSex")
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
decision_tree = decision_tree.fit(iris.data, iris.target)
r = export_text(decision_tree, feature_names=iris['feature_names'])
print(r)

X2, y2 = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
     % (X_test.shape[0], (y_test != y_pred).sum()))
#print(graph.render("iris"))

breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
X = X[['mean area', 'mean compactness']]
y = pd.Categorical.from_codes(breast_cancer.target, breast_cancer.target_names)
y = pd.get_dummies(y, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

sns.scatterplot(
    x='mean area',
    y='mean compactness',
    hue='benign',
    data=X_test.join(y_test, how='outer')
)

plt.scatter(
    X_test['mean area'],
    X_test['mean compactness'],
    c=y_pred,
    cmap='coolwarm',
    alpha=0.7
)
print(confusion_matrix(y_test, y_pred))
data=pd.read_csv("iris.csv")

print(iris.data[:3])
print(iris.data[15:18])
print(iris.data[37:40])

X = iris.data[:, (2, 3)]

print(iris.target)

y = (iris.target==0).astype(np.int8)
print(y)

p = Perceptron(random_state=42,
              max_iter=10,
              tol=0.001)
p.fit(X, y)


values = [[1.5, 0.1], [1.8, 0.4], [1.3,0.2]]

for value in X:
    pred = p.predict([value])
    print([pred])

X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
y = [0, 0, 0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

print(clf.fit(X, y))

X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
y = [0, 0, 0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

print(clf.fit(X, y))
# creating an classifier from the model:
datasets = train_test_split(iris.data, iris.target,
                            test_size=0.2)

train_data, test_data, train_labels, test_labels = datasets

scaler = StandardScaler()

# we fit the train data
scaler.fit(train_data)

# scaling the train data
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

print(train_data[:3])

mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)

# let's fit the training data to our model
mlp.fit(train_data, train_labels)

predictions_train = mlp.predict(train_data)
print(accuracy_score(predictions_train, train_labels))
predictions_test = mlp.predict(test_data)
print(accuracy_score(predictions_test, test_labels))

confusion_matrix(predictions_train, train_labels)

confusion_matrix(predictions_test, test_labels)

print(classification_report(predictions_test, test_labels))