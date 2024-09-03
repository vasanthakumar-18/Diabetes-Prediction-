'''



------------------------------------------------------------------------------------------------------------------------------------------


	PROJECT NAME    :       DIABETES PREDICTION

	TEAM NAME       :       TENSORFLOW
   
	TEAM MEMBERS    :       1.ELANGOVAN B
                                2.KALAIVANAN V
			        3.RAMESH M
                                4.VASANTHAKUMAR G

	FILE NAMES      :       diabetes_prediction.py,diabetes_prediction.csv,diabetes_prediction.pdf

	PROGRAM         :       SUPERVISED LEARNING AND SUPPORT VECTOR MACHINE(SVM)


------------------------------------------------------------------------------------------------------------------------------------------




'''

#IMPORTING THE LIBRARIES
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#DATA COLLECTION AND ANALYSIS
diabetes_dataset = pd.read_csv("FILE PATH")
# print the first 5 rows of the dataset
diabetes_dataset.head(5)
# number of rows and Columns in this dataset
diabetes_dataset.shape
# getting the statistical measures of the data
diabetes_dataset.describe()
#getting column from the outcome column(i.e label)
diabetes_dataset['Outcome'].value_counts()
#0--->Non-Diabetes
#1--->Diabetes
#getting mean value for the Outcomes
mean_values = diabetes_dataset.groupby('Outcome').mean()
print(mean_values)
#separating the data and labels
x=diabetes_dataset.drop(columns='Outcome',axis=1) 
y=diabetes_dataset['Outcome']
print(x)
print(y)


#DATA STANDARDIZATION
#create a variable scalar and load the StandardScalar function 
scalar = StandardScaler()
#fitting the inconsistent data with our standard scalar function
scalar.fit(x)
standardized_data=scalar.transform(x)
print(standardized_data)
#assign the standardized_data again with x 
x=standardized_data
y=diabetes_dataset['Outcome']
print(x)
print(y)


#TRAIN TEST SPLIT
#x-data
#y-label
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
print(x.shape,x_train.shape,x_test.shape)


#TRAINING THE MODEL
# create a var classifier and load the algorithms
classifier1=svm.SVC(kernel='linear')
classifier2 = KNeighborsClassifier(n_neighbors=5)
classifier3 = DecisionTreeClassifier()
classifier4= GaussianNB() 
#training the classifier models 
classifier1.fit(x_train,y_train)
classifier2.fit(x_train,y_train)
classifier3.fit(x_train,y_train)
classifier4.fit(x_train,y_train)


#MODEL EVALUATION
#ACCURACY SCORE ON THE TRAINING DATA(SVM)
x_train_prediction=classifier1.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print("Accuracy score of the training data:",training_data_accuracy)

#ACCURACY SCORE ON THE TRAINING DATA(KNN)
x_train_prediction=classifier2.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print("Accuracy score of the training data:",training_data_accuracy)

#ACCURACY SCORE ON THE TRAINING DATA(Decision Tree)
x_train_prediction=classifier3.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print("Accuracy score of the training data:",training_data_accuracy)

#ACCURACY SCORE ON THE TRAINING DATA(Naive Bayes)
x_train_prediction=classifier4.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print("Accuracy score of the training data:",training_data_accuracy)

#ACCURACY SCORE ON THE TESTING DATA(SVM)
x_test_prediction=classifier1.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print("Accuracy score of the test data:",test_data_accuracy)

#ACCURACY SCORE ON THE TESTING DATA(KNN)
x_test_prediction=classifier2.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print("Accuracy score of the test data:",test_data_accuracy)

#ACCURACY SCORE ON THE TESTING DATA(Decision Tree)
x_test_prediction=classifier3.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print("Accuracy score of the test data:",test_data_accuracy)

#ACCURACY SCORE ON THE TESTING DATA(Naive Bayes)
x_test_prediction=classifier4.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print("Accuracy score of the test data:",test_data_accuracy)

#MAKING A PREDICTIVE SYSTEM
input_data=[1,23,45,2.45,0,0.87,3,45]
#changing the input_data to numpy array
input_data_np_array=np.asarray(input_data)
#reshape the array as we predicting for one instance
input_data_reshape=input_data_np_array.reshape(1,-1)
#standardize the input data
std_data=scalar.transform(input_data_reshape)
print(std_data)

prediction=classifier.predict(std_data)
print(prediction1)


if(prediction[0]==0):
   print("The person is not diabetic")
else:
   print("The person is diabetic")
