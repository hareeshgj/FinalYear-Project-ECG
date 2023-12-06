from deepBeliefNetworkMaster.dbn import SupervisedDBNRegression
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold,train_test_split


def get_score(model,X_train,X_test,y_train,y_test): #this function is used to check the accuracy score for a given model, training and testing data
	model.fit(X_train,y_train)
	return model.score(X_test,y_test)

kf=KFold(n_splits=10)

train_y = [] #Actual result of the data used in testing of the valence 
train_a = [] #Actual result of the data used in testing of the arousal

train_x = np.genfromtxt('train.csv',delimiter=',')
train_x = np.array(train_x)
train_x=train_x.astype(np.long)
train_x=train_x.transpose()
f = open("labels_0.dat","r")


for i in f:
	train_y.append(i) #copying data from the file to the list
train_y = np.array(train_y).astype(np.float)
train_y = train_y.astype(np.int)#changing the list to numpy array and its value type from float to int


clf = SupervisedDBNRegression(hidden_layers_structure=[100], learning_rate_rbm=0.05,learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,batch_size=100, activation_function='relu')

for train_index,test_index in kf.split(train_x):
	X_train,X_test,y_train,y_test=train_x[train_index],train_x[test_index],train_y[train_index],train_y[test_index]	

clf.fit(X_train, y_train)
predicted_val = clf.predict(X_test)
print('Valence Mean Squared Error: ' +str(metrics.mean_squared_error(y_test, predicted_val))+'\n')
print('Valence r^2: ' +str(metrics.r2_score(y_test, predicted_val))+'\n')
f = open("labels_1.dat","r")
for i in f:
	train_a.append(i) #copying data from the file to the list
train_a = np.array(train_a).astype(np.float)
train_a = train_a.astype(np.int) #changing the list to numpy array and its value type from float to int


kf1=KFold(n_splits=10)
clfar = SupervisedDBNRegression(hidden_layers_structure=[100], learning_rate_rbm=0.05,learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,batch_size=100, activation_function='relu')

for train_index,test_index in kf1.split(train_x):
	X_train1,X_test1,y_train1,y_test1=train_x[train_index],train_x[test_index],train_a[train_index],train_a[test_index]

clfar.fit(X_train1, y_train1)
arousal_val = clfar.predict(X_test1)
print('Arousal Mean Squared Error: ' +str(metrics.mean_squared_error(y_test1, arousal_val))+'\n')
print('Arousal r^2: ' +str(metrics.r2_score(y_test1, arousal_val))+'\n')


print(arousal_val[0])
print(predicted_val[0])
if arousal_val[0]<5 and predicted_val[0]>0:
    print("Emotion : Relaxed")
elif arousal_val[0]>5 and predicted_val[0]>0:
    print("Emotion : Happy")
elif arousal_val[0]>5 and predicted_val[0]<0:
    print("Emotion : Angry")
else:
    print("Emotion : Bored")
