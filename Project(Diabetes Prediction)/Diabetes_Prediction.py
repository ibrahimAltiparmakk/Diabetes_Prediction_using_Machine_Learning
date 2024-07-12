# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 19:36:57 2024

@author: sarib
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score # accuracy_score fonksiyonu, sklearn.metrics modülünün bir parçasıdır ve sınıflandırma problemlerinde bir modelin performansını ölçmek için kullanılır. Bu fonksiyon, bir modelin tahmin ettiği sınıflar ile gerçek sınıflar arasındaki doğruluk oranını (accuracy) hesaplar.
# Accuracy = doğru tahmin sayısı / toplam tahmin sayısı , olarak hesaplanır 1 olursa bütün tahmniler doğrudur


diabetes_dataset = pd.read_csv('diabetes.csv')
print(diabetes_dataset.head()) # ilk 5 satırı head foknsiyonuyla yazdırabiliriz

print(diabetes_dataset.shape) # datasetin kaç satır ve kaç sutundan oluştuğunu söyler

print(diabetes_dataset['Outcome'].value_counts()) # Outcome sutununda 0 dan kaç tane , 1 den kaç tane onu görürüz
# 0 => diyabet olmayan
# 1 => diyabet olan

print(diabetes_dataset.groupby('Outcome').mean())
Outcome_mean=diabetes_dataset.groupby('Outcome').mean()
'''
diabetes_dataset.groupby('Outcome').mean() kodu, pandas DataFrame'in 'Outcome' sütununa göre gruplanmasını 
ve her grubun ortalama değerlerinin hesaplanmasını sağlar. Bu, her sınıf için (diyabetli ve diyabetli olmayan)
diğer sütunların ortalama değerlerini görmenize olanak tanır.
groupby('Outcome'): Veri setini 'Outcome' sütunundaki değerlere göre gruplar.
.mean(): Her grup için tüm diğer sütunların ortalama değerlerini hesaplar.
'''

X = diabetes_dataset.drop(columns = 'Outcome')# drop fonksiyonu, pandas DataFrame'den belirtilen etiketlere (satır veya sütun) sahip ögeleri kaldırmak için kullanılır.
Y = diabetes_dataset['Outcome']

#---------------- standartlaştırma -----------------
scaler = StandardScaler()
standardized_data = scaler.fit_transform(X)

X = standardized_data

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
# stratify=Y: Y değişkenindeki sınıf oranları, hem eğitim hem de test setinde korunur
'''
Diyelim ki Y hedef değişkeni şu sınıf dağılımına sahip:

Sınıf 0: %70
Sınıf 1: %30
stratify parametresi kullanılarak yapılan bir ayırma, hem eğitim hem de test setinde bu sınıf oranlarını koruyacaktır.
'''

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train) 

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print("Train verilerinin doğruk oranı" , training_data_accuracy) # doğruluk oranı

test_predict = classifier.predict(X_test)
test_data_accuracy = accuracy_score(test_predict,Y_test)
print("Test verilerinin doğruluk oranı", test_data_accuracy)

 
input_data = (4,110,92,0,0,37.6,0.191,30) # datasetten örnek bir veri verelim

input_data_as_numpy_array = np.asarray(input_data) # Bu adımda, input_data tuple'ı bir NumPy dizisine (numpy array) dönüştürülür. NumPy dizileri, bilimsel hesaplamalar ve veri manipülasyonu için verimli veri yapılarıdır.
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1) # Bu adımda, input_data_as_numpy_array yeniden şekillendirilir. reshape(1, -1) ifadesi, veriyi tek bir örnek (satır) ve çoklu özellikler (sütunlar) şeklinde yeniden düzenler. Bu, modelin tahmin fonksiyonuna (predict) uygun hale getirilmesi için gereklidir.
#reshape de ilk kısma kaç satır olacağı , ikinci kısma kaç sutun olacağı yazılır -1 yazarsan otomatik sutun sayısını buluyor
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction) # doğru tahmin etti

if (prediction == 0):
    print("Diyabet hastası değil")
else:
    print("Diyabet hastası")