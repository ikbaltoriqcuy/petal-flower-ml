import pandas as pd
import sys
import sklearn.neighbors.typedefs
import sklearn.utils._cython_blas
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
import tornado

# get data agaricus-lepiota.csv
dataSetsMushroom = pd.read_csv(r'./agaricus-lepiota.csv')

#processing encode
le = preprocessing.LabelEncoder()

#get label from file csv
label = le.fit_transform(dataSetsMushroom['a'].values)

#get attribute from file csv
cap_shape                   = le.fit_transform(dataSetsMushroom['b'].values)
cap_surface                 = le.fit_transform(dataSetsMushroom['c'].values)
cap_color                   = le.fit_transform(dataSetsMushroom['d'].values)
bruises                     = le.fit_transform(dataSetsMushroom['e'].values)
odor                        = le.fit_transform(dataSetsMushroom['f'].values)
gill_attachment             = le.fit_transform(dataSetsMushroom['g'].values)
gill_spacing                = le.fit_transform(dataSetsMushroom['h'].values)
gill_size                   = le.fit_transform(dataSetsMushroom['i'].values)
gill_color                  = le.fit_transform(dataSetsMushroom['j'].values)
stalk_shape                 = le.fit_transform(dataSetsMushroom['k'].values)
stalk_root                  = le.fit_transform(dataSetsMushroom['l'].values)
stalk_surface_above_ring    = le.fit_transform(dataSetsMushroom['m'].values)
stalk_surface_below_ring    = le.fit_transform(dataSetsMushroom['n'].values)
stalk_color_above_ring      = le.fit_transform(dataSetsMushroom['o'].values)
stalk_color_below_ring      = le.fit_transform(dataSetsMushroom['p'].values)
veil_type                   = le.fit_transform(dataSetsMushroom['q'].values)
veil_color                  = le.fit_transform(dataSetsMushroom['r'].values)
ring_number                 = le.fit_transform(dataSetsMushroom['s'].values)
ring_type                   = le.fit_transform(dataSetsMushroom['t'].values)
spore_print_color           = le.fit_transform(dataSetsMushroom['u'].values)
population                  = le.fit_transform(dataSetsMushroom['v'].values)
habitat                     = le.fit_transform(dataSetsMushroom['w'].values)

#combinig weather and temp into single listof tuples
features=list(zip(cap_shape,cap_surface,cap_color,
                  bruises,odor,gill_attachment,
                  gill_spacing,gill_size,gill_color,
                  stalk_shape,stalk_root,stalk_surface_above_ring,
                  stalk_surface_below_ring,stalk_color_above_ring,
                  stalk_color_below_ring,veil_type,veil_color,
                  ring_number,ring_type,spore_print_color,
                  population,habitat
                  )
              )

model = KNeighborsClassifier(n_neighbors=5)

#Predict Output
# model.fit(features,label)
# predicted= model.predict([[4,2,0,1,5,2,1,0,2,1,3,3,0,6,4,0,1,1,2,3,0,5]])
# print(predicted)

#Train and test data with epoch 80:20

x_train, x_test,\
y_train, y_test = train_test_split(features,label,train_size=0.8,test_size=0.2,random_state=0)
print("Train and test data with epoch 80:20 \n")
print("Data train : ",x_train)
print("Data test : ",x_test)
print("")

model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Hasil Train Data = ", y_pred)  #data train
print("Hasil Test Data  = ", y_test)  #data test

print("akurasi skor = ", metrics.accuracy_score(y_test, y_pred))

plt.title("Hasil prediksi data training terhadap data test")
plt.plot(y_test, y_pred)
plt.xlabel('Hasil Test data')
plt.ylabel('Hasil Train data')
plt.show()
input("")