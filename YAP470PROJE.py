#YAP470 PROJE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

#Öznitelik seçimi için deneme yapacağım, data => oynanmamış veriseti
data = pd.read_csv('breast-cancer.csv') #csv dosyası .py veya .ipynb ile aynı klasörde olmalı
data.replace(to_replace='M', value = 0, inplace=True) 
data.replace(to_replace='B', value = 1, inplace=True) #azinlik, positive class'imiz kotu huylu tumorler, bunu dogru tahmin etmek daha onemli
data.drop(columns=['id'], inplace=True) #id satirini cikariyoruz, tibbi bir veri degil, bulunmasi siniflandiricinin kafasini karistirir

#data_relevant => özniteliklerin korelasyonu ile oluşturulmuş veriseti
korelasyon = data.corr()
korelasyon_degerleri = abs(korelasyon["diagnosis"])
oznitelikler_relevant =  korelasyon_degerleri[korelasyon_degerleri > 0.2] #korelasyonu yüksek olanları ayır
oznitelikler_relevant = list(oznitelikler_relevant.index) #sütunların isimlerini extract et
oznitelikler_relevant.remove('diagnosis') #target sütunun verisetinden ayır
data_relevant = data[oznitelikler_relevant] #target ile korelasyonu en yüksek özniteliklerden oluşan bir dataframe
target = data['diagnosis'].values #target sutunu

data.drop(columns=['diagnosis'], inplace=True) #target sütununu verisetinden ayır

#SVM kullanılacağı için öznitelik ölçeklemesi, outlier yok ve dağılım uygun olduğu için standart scaler kullanıldı:
data_relevant = StandardScaler().fit_transform(data_relevant)
data_relevant = pd.DataFrame(data_relevant)

'''
Eğitim, test ve cross-validation verilerini ayirma, %60 training, %20 test, %20 cross-validation
data_relevant_train_val, data_relevant_test, target_train_val, target_test = train_test_split(data_relevant, target, test_size=0.2, random_state=50)
data_relevant_train, data_relevant_val, target_train, target_val = train_test_split(data_relevant_train_val, target_train_val, test_size=0.25, random_state=50)
ILK BOYLE YAPTIM AMA GRİD SEARCH İÇİN DEĞİŞTİRDİM, CROSS VALİDATİON ZATEN ORADA YAPILDI
'''

#Training ve test için data split
data_relevant_train, data_relevant_test, target_train, target_test = train_test_split(data_relevant, target, test_size=0.25, random_state=50) #50 ihtiyari seçildi, bir daha kullacağı zaman yine 50 olsa yeter

cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=50) #5-fold çok yaygın, target değerler dengesiz olduğu için stratified k-fold kullandım

svc = SVC()
svc_parametreleri = {
    'C': [0.1, 1, 10, 100, 1000], #deney yap hep aynı sonuc
    'gamma': [10, 1, 0.1, 0.01, 0.001], #deney yap hep aynı sonuc
    'kernel': ['rbf','sigmoid','linear'] }
#kernel olarak Radial Basis Function, dersten biliyorum ve yaygın. Sigmoid: non-linear sınıflandırma EDA'dan gereksiz olduğu bulduk ancak işlemi çok yavaşlatmıyor
#laplacian rbf ile de deneyecektim yoktu
#EDA'dan linear'i bulduk, yine de rbf'i seçti
grid_search_svc = GridSearchCV(estimator=svc,
                               param_grid=svc_parametreleri,
                               cv=cross_validation,
                               verbose=1,
                               scoring='roc_auc')

grid_search_relevant = grid_search_svc.fit(data_relevant_train, target_train)
grid_search_relevant.best_params_

#İşlenmiş veriler ile training ve test
svm_classifier = svc.set_params(**grid_search_relevant.best_params_)
svc.fit(data_relevant_train, target_train)
svm_prediction_relevant = svc.predict(data_relevant_test)

print(classification_report(target_test, svm_prediction_relevant))
print(confusion_matrix(target_test, svm_prediction_relevant))
print(f'ROC-AUC score : {roc_auc_score(target_test, svm_prediction_relevant)}')
print(f'Accuracy score : {accuracy_score(target_test, svm_prediction_relevant)}')

#ölçeklenmemiş veriler ile training ve test
oznitelikler = list(data.index) #sütunların isimlerini extract et
data_to_train = data[oznitelikler] 

data_train, data_test, target_train, target_test = train_test_split(data_to_train, target, test_size=0.25, random_state=50)

grid_search_svc = GridSearchCV(estimator=svc,
                               param_grid=svc_parametreleri,
                               cv=cross_validation,
                               verbose=1,#neyin seçildiğinin çıktısını veriyor
                               scoring='roc_auc')#yaygın bir ölçüt

grid_search = grid_search_svc.fit(data_train, target_train)
grid_search_relevant.best_params_
svm_classifier = svc.set_params(**grid_search_relevant.best_params_)
svc.fit(data_train, target_train)
svm_prediction = svc.predict(data_test)

print(classification_report(target_test, svm_prediction_relevant))
print(confusion_matrix(target_test, svm_prediction_relevant))
print(f'ROC-AUC score : {roc_auc_score(target_test, svm_prediction_relevant)}')
print(f'Accuracy score : {accuracy_score(target_test, svm_prediction_relevant)}')

#random forest
random_forest = RandomForestClassifier(random_state=50)
random_forest_parametreleri = {
    'min_samples_leaf': range(1, 5),
    'max_features': ['sqrt', 'log2']
}
#'sqrt' rastgelelik ve overfit arasında güzel bir denge kuruyor
#'integer' veya 'float' koymak istemedim, en uygununu grid search bulsun diye
#'None' koymak da random forest'ın felsefesine aykırı
grid_search_randomforest = GridSearchCV(estimator=random_forest,
                               param_grid=random_forest_parametreleri,
                               cv=cross_validation,
                               verbose=1,
                               scoring='roc_auc')

grid_result_rfc = grid_search_randomforest.fit(data_relevant_train, target_train)
grid_result_rfc.best_params_

random_forest = random_forest.set_params(**grid_result_rfc.best_params_)
random_forest.fit(data_relevant_train, target_train)
random_forest_sonuc = random_forest.predict(data_relevant_test)

print(classification_report(target_test, random_forest_sonuc))
print(confusion_matrix(target_test, random_forest_sonuc))
print(f'ROC-AUC score : {roc_auc_score(target_test, random_forest_sonuc)}')
print(f'Accuracy score : {accuracy_score(target_test, random_forest_sonuc)}')
