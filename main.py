from sklearn.datasets import load_digits #inbuilt iris dataset from scikit learn
from sklearn.model_selection import train_test_split #for spliting dataset for training and testing
from sklearn import metrics #for finding accuracy score of model in this program
import joblib #for storing the model as a joblib file to avoid unwanted training steps
import skimage,sklearn
from sklearn.naive_bayes import GaussianNB



digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25,random_state=1)
classifier =GaussianNB()
classifier.fit(X_train, y_train)

score = classifier.score(X_test, y_test)
print("Accuracy:", score)
joblib.dump(classifier,'hwdr.joblib')