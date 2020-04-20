import pandas as pd
import csv
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.metrics import accuracy_score  # Accuracy
from sklearn.tree import DecisionTreeClassifier  # Decisition Tree Classifier
from sklearn.model_selection import train_test_split  # split data set
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics

with open('crime.csv', 'r') as csvfile:
    data = csv.reader(csvfile, delimiter=',', quotechar='|')
    dataCl = []
    dataCl2 = []
    rangeClass1 = 0
    rangeClass2 = 0
    skipRow = False

    for row in data:
        for column in row:
            if column == '' or column == ' ' or column == None:
                skipRow = True
                break
            else:
                dataCl.append(row)
        if skipRow == True:
            continue

    for row in dataCl:
        if row[-1] == '3501':
            rangeClass1 += 1
            if rangeClass1 <= 500:
                dataCl2.append(row)
        if row[-1] == '706':
            rangeClass2 += 1
            if rangeClass2 <= 500:
                dataCl2.append(row)


def cleanData(data):
    data = pd.DataFrame(data)
    sizeData = len(data)
    sizeDataRow = data.shape[1]
    features = []
    clases = []
    row = []
    for i in range(sizeData):
        row = []
        for e in range(6):
            row.append(float(data.loc[i, e]))
        features.append(row)
        clases.append(int(data.loc[i, 6]))
    return features, clases


dataFeatures, dataClass = cleanData(dataCl2)


featuresTrain, featuresTest, classTrain, classTest = train_test_split(
    dataFeatures, dataClass, test_size=0.2, random_state=0)

############ Classifier KNN ##########

classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(featuresTrain, classTrain)
classifier.score(featuresTest, classTest)


print(classifier.predict([[120, 8, 13, 42.3542716, -71.0683861, 1]]))  # 3501
print(classifier.predict([[334, 8, 14, 42.3110655, -71.0669323, 7]]))  # 3301
print(classifier.predict([[121, 7, 7, 42.3494286, -71.0653664, 2]]))  # 706

print(featuresTest)

############ Classifier DecisionTree ##########

clasifierDecisiontree = DecisionTreeClassifier()
clasifierDecisiontree.fit(featuresTrain, classTrain)
clasifierDecisiontree.score(featuresTest, classTest)

print(clasifierDecisiontree.predict([[121, 7, 7, 42.3494286, -71.0653664, 2]]))  # 706
print(clasifierDecisiontree.predict([[120, 8, 13, 42.3542716, -71.0683861, 1]]))  # 3501
############ Classifier Logistic Regression ##########

classifierLogR = LogisticRegression()
classifierLogR.fit(featuresTrain, classTrain)
classifierLogR.score(featuresTest, classTest)

print(classifierLogR.predict([[121, 7, 7, 42.3494286, -71.0653664, 2]]))  # 706
print(classifierLogR.predict([[120, 8, 13, 42.3542716, -71.0683861, 1]]))  # 3501

############ Classifier support vector machine ##########

clasifierSupVec = svm.SVC(gamma='scale')
clasifierSupVec.fit(featuresTrain, classTrain)
clasifierSupVec.score(featuresTest, classTest)
print(clasifierSupVec.predict(
    [[121, 7, 7, 42.3494286, -71.0653664, 2]]))  # 706 test
print(clasifierSupVec.predict(
    [[120, 8, 13, 42.3542716, -71.0683861, 1]]))  # 3501 test
