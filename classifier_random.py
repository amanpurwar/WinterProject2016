from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.ensemble import EasyEnsemble
from imblearn.over_sampling import SMOTE


#create model
clf_easy = RandomForestClassifier(n_estimators=90,min_samples_split=1,max_features='sqrt')
clf_smote = RandomForestClassifier(n_estimators=90,min_samples_split=1,max_features='sqrt')
#for easy ensemble
#fit the model according to the training data.
ee=EasyEnsemble()
X_easy,y_easy=ee.fit_sample(X,y)
clf_easy.fit(X_easy, y_easy)

#predict labels for heldout data
predictedy = clf_easy.predict(heldoutX)

#print precision recall table
print(classification_report(heldouty, predictedy))

#report accuracy on heldout data
print "Accuracy= ",clf_easy.score(heldoutX, heldouty)

#for SMOTE
#fit the model according to the training data.
smte=SMOTE()
X_smote,y_smote=smte.fit_sample(X,y)
clf_smote.fit(X_smote, y_smote)

#predict labels for heldout data
predictedy = clf_smote.predict(heldoutX)

#print precision recall table
print(classification_report(heldouty, predictedy))

#report accuracy on heldout data
print "Accuracy= ",clf_smote.score(heldoutX, heldouty)
