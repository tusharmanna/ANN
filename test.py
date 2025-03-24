import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

#load dataset
data = pd.read_csv('C:\work\learnai\ANN\Churn_Modelling.csv')
#print(data.head())

#drop irrelevant columns such as rownumber,customerid,surname
data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
#print(data.head())

label_encoder_gender=LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])

#use onehoty encoder for geography column
one_hot_encoder_geo = OneHotEncoder()
data_geo = one_hot_encoder_geo.fit_transform(data[['Geography']])
#data_geo.toarray()
data_geo = pd.DataFrame(data_geo.toarray(), columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))
data = pd.concat([data, data_geo], axis=1)
#data.head()
data.drop(['Geography'], axis=1, inplace=True)
data.head()

#divide the dataste into independent and dependent features
X=data.drop('Exited', axis=1)
y=data['Exited']

#split the dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#feature scaling
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#save scaler into pickle format
scaler_file = 'scaler.pkl'
pickle.dump(sc, open(scaler_file, 'wb'))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import datetime

#build ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#setupo tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir)]

#setup early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5,restore_best_weights=True)
callbacks.append(early_stopping)

#train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks)


#save the model
model.save('ann_model.h5')





#load the model
model = keras.models.load_model('ann_model.h5')

#load encoders and scalar
sc = pickle.load(open('scaler.pkl', 'rb'))
label_encoder_gender = pickle.load(open('label_encoder_gender.pkl', 'rb'))
one_hot_encoder_geo = pickle.load(open('one_hot_encoder_geo.pkl', 'rb'))

#example input data
example_data = {
    'CreditScore': 600,
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 1,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'Geography': 'France'
}

#preprocess example data
example_data = pd.DataFrame([example_data])
example_data['Gender'] = label_encoder_gender.transform(example_data['Gender'])

#use label_encoder_geo to transform data
example_data['Geography'] = label_encoder_geo.transform(example_data['Geography'])



#use one_hot_encoder_geo to transform data
example_data = pd.concat([example_data, pd.get_dummies(example_data['Geography'])], axis=1)
example_data.drop('Geography', axis=1, inplace=True)

#scale the data
example_data = sc.transform(example_data)

#make predictions
y_pred = model.predict(example_data)

if y_pred > 0.5:
    print('Customer will leave the bank')
else:
    print('Customer will not leave the bank')

#save the predictions into a csv file
pd.DataFrame(y_pred, columns=['Exited']).to_csv('y_pred.csv', index=False)

#evaluate the model
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

#close tensorboard
os.system('taskkill /F /IM python.exe')










