from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from imblearn.ensemble import EasyEnsemble
from imblearn.over_sampling import SMOTE
#create model
clf3_easy = RandomForestClassifier(n_estimators=90,min_samples_split=1,max_features='sqrt')
clf3_smote=RandomForestClassifier(n_estimators=90,min_samples_split=1,max_features='sqrt')
'''
increasing n_estimators increases accuracy as we are seeing more number of trees in the forest

min_samples_split or minimum number of samples required to split an internal node, when set to 1, leads to increased accuracy

max_features include number of features to consider when looking for the best split
'''
#Evaluate a score by cross-validation using east ensemeble
ee=EasyEnsemble()
X_easy,y_easy=ee.fit_sample(X,y)
scores3_easy = cross_val_score(clf3_easy, X_easy, y_easy, cv=10)
#Evaluate a score by cross-validation using SMOTE
smte=SMOTE()
X_smote,y_smote=smte.fit_sample(X,y)
scores3_smote = cross_val_score(clf3_smote, X_smote, y_smote, cv=10)

#report accuracy
print("Accuracy for Random Forest Classifier using easy ensesmble: %0.2f (+/- %0.2f)" % (scores3_easy.mean(), scores3_easy.std() * 2))
print("Accuracy for Random Forest Classifier using SMOTE: %0.2f (+/- %0.2f)" % (scores3_smote.mean(), scores3_smote.std() * 2))
