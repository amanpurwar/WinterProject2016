from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from imblearn.ensemble import EasyEnsemble
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.ensemble import BalanceCascade
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFpr
from sklearn.datasets import make_classification
from collections import Counter
from imblearn.ensemble import BalanceCascade
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

file = open("Data/EssentialsofML/trainingFiltered.csv")
file.readline()

#get unigrams list of word
def getUnigrams(word):
    return list(word) 

#get bigrams list of word
def getBigrams(word):
    return [ word[i]+word[i+1] for i in range(0,len(word)-1)]

#get trigrams list of word
def getTigrams(word):
    return [ word[i]+word[i+1]+word[i+2] for i in range(0,len(word)-2)]

#read columns word1, word2 and labels
word1Vec = []
word2Vec = []
labelVec = []

for line in file.readlines():
    splited_line = line.split(',')
    word1Vec.append(splited_line[1])
    word2Vec.append(splited_line[2])
    labelVec.append(splited_line[3].strip())
    
#enumurate unigrams, bigrams and trigrams for columns word1 and word2
unigramWord1 = reduce(lambda a,b: list(set(a+b)),map(getUnigrams,word1Vec))
bigramWord1 = reduce(lambda a,b: list(set(a+b)),map(getBigrams,word1Vec))
trigramWord1 = reduce(lambda a,b: list(set(a+b)),map(getTigrams,word1Vec))

unigramWord2 = reduce(lambda a,b: list(set(a+b)),map(getUnigrams,word2Vec))
bigramWord2 = reduce(lambda a,b: list(set(a+b)),map(getBigrams,word2Vec))
trigramWord2 = reduce(lambda a,b: list(set(a+b)),map(getTigrams,word2Vec))

print "10 unigams for word1 : ","\t".join(unigramWord1[:10])
print "10 bigams  for word1 : ","\t".join(bigramWord1[:10])
print "10 trigams for word1 : ","\t".join(trigramWord1[:10])
print "10 unigams for word2 : ","\t".join(unigramWord2[:10])
print "10 bigams  for word2 : ","\t".join(bigramWord2[:10])
print "10 trigams for word2 : ","\t".join(trigramWord2[:10])
    
#combining ngrmas for word1 and word2
featureVecForWord1 = unigramWord1+bigramWord1+trigramWord1
featureVecForWord2 = unigramWord2+bigramWord2+trigramWord2



import math

#count of all feactures over column word1, word2 
countNgramsWord1 = [ sum([ word1.count(ngram) for word1 in word1Vec ]) for ngram in featureVecForWord1]
countNgramsWord2 = [ sum([ word2.count(ngram) for word2 in word2Vec ]) for ngram in featureVecForWord2]

#for each class constructing a vector of counts of occurances for each feature
aCountWord1 = [ sum([ word1Vec[i].count(ngram) for i in range(len(word1Vec)) if labelVec[i]=='A' ]) for ngram in featureVecForWord1 ]
kCountWord1 = [ sum([ word1Vec[i].count(ngram) for i in range(len(word1Vec)) if labelVec[i]=='K' ]) for ngram in featureVecForWord1 ]
bCountWord1 = [ sum([ word1Vec[i].count(ngram) for i in range(len(word1Vec)) if labelVec[i]=='B' ]) for ngram in featureVecForWord1 ]
tCountWord1 = [ sum([ word1Vec[i].count(ngram) for i in range(len(word1Vec)) if labelVec[i]=='T' ]) for ngram in featureVecForWord1 ]
dCountWord1 = [ sum([ word1Vec[i].count(ngram) for i in range(len(word1Vec)) if labelVec[i]=='D' ]) for ngram in featureVecForWord1 ]

aCountWord2 = [ sum([ word2Vec[i].count(ngram) for i in range(len(word2Vec)) if labelVec[i]=='A' ]) for ngram in featureVecForWord2 ]
kCountWord2 = [ sum([ word2Vec[i].count(ngram) for i in range(len(word2Vec)) if labelVec[i]=='K' ]) for ngram in featureVecForWord2 ]
bCountWord2 = [ sum([ word2Vec[i].count(ngram) for i in range(len(word2Vec)) if labelVec[i]=='B' ]) for ngram in featureVecForWord2 ]
tCountWord2 = [ sum([ word2Vec[i].count(ngram) for i in range(len(word2Vec)) if labelVec[i]=='T' ]) for ngram in featureVecForWord2 ]
dCountWord2 = [ sum([ word2Vec[i].count(ngram) for i in range(len(word2Vec)) if labelVec[i]=='D' ]) for ngram in featureVecForWord2 ]

def entropy(x):
    return -(x*math.log(x))

def get_prob_vec(tagCountWordx,countNgramsWordx,featureVecForWordx):
    return [ (tagCountWordx[i]+1)/(float(countNgramsWordx[i])+len(featureVecForWordx)) for i in range(len(tagCountWordx)) ]

#calculating probabilities and applying p(x)*log(p(x))
aCountWord1 = map(entropy, get_prob_vec(aCountWord1, countNgramsWord1, featureVecForWord1))
kCountWord1 = map(entropy, get_prob_vec(kCountWord1, countNgramsWord1, featureVecForWord1))
bCountWord1 = map(entropy, get_prob_vec(bCountWord1, countNgramsWord1, featureVecForWord1))
tCountWord1 = map(entropy, get_prob_vec(tCountWord1, countNgramsWord1, featureVecForWord1))
dCountWord1 = map(entropy, get_prob_vec(dCountWord1, countNgramsWord1, featureVecForWord1))

aCountWord2 = map(entropy, get_prob_vec(aCountWord2, countNgramsWord2, featureVecForWord2))
kCountWord2 = map(entropy, get_prob_vec(kCountWord2, countNgramsWord2, featureVecForWord2))
bCountWord2 = map(entropy, get_prob_vec(bCountWord2, countNgramsWord2, featureVecForWord2))
tCountWord2 = map(entropy, get_prob_vec(tCountWord2, countNgramsWord2, featureVecForWord2))
dCountWord2 = map(entropy, get_prob_vec(dCountWord2, countNgramsWord2, featureVecForWord2))

#calculating entropy for each feature
ngramEntropyWord1 = [ aCountWord1[i] + kCountWord1[i] + bCountWord1[i] + tCountWord1[i] + dCountWord1[i] for i in range(len(featureVecForWord1))]
ngramEntropyWord2 = [ aCountWord2[i] + kCountWord2[i] + bCountWord2[i] + tCountWord2[i] + dCountWord2[i] for i in range(len(featureVecForWord2))]

#taking top 1000 features for both word1 and word2
topIndices1 = sorted(range(len(ngramEntropyWord1)), key=lambda i: ngramEntropyWord1[i])[-1000:]
topIndices2 = sorted(range(len(ngramEntropyWord2)), key=lambda i: ngramEntropyWord2[i])[-1000:]

#list containing best 1000 features for word1 and word2
colfor1 = [featureVecForWord1[i] for i in topIndices1]
colfor2 = [featureVecForWord2[i] for i in topIndices2]

print 'Best 10  features for word1 : ',"\t".join(colfor1[:10])
print 'Worst 10 features for word1 : ',"\t".join(colfor1[-10:])
print 'Best 10  features for word2 : ',"\t".join(colfor2[:10])
print 'Worst 10 features for word2 : ',"\t".join(colfor2[-10:])



def calculate_features(inFileName,outFileName):
    word1Vec = []
    word2Vec = []
    remainingCols = []

    file = open(inFileName)
    outFile = open(outFileName,"w")

    outFile.write(file.readline().strip('\n')[1:]+','+','.join(colfor1)+','+','.join(colfor2)+"\n")

    for line in file.readlines():
        splited_line = line.split(',')
        word1Vec.append(splited_line[1])
        word2Vec.append(splited_line[2])
        remainingCols.append(','.join(splited_line[3:]).split('\n')[0])

    ngramCountWord1 = [[ word1.count(ngram) for ngram in colfor1 ] for word1 in word1Vec ]
    ngramCountWord2 = [[ word2.count(ngram) for ngram in colfor2 ] for word2 in word2Vec ]

    for i in range(len(word1Vec)):
        outFile.write(str(word1Vec[i])+','+str(word2Vec[i])+','+str(remainingCols[i])+','+','.join(map(str,ngramCountWord1[i]))+','+','.join(map(str,ngramCountWord2[i]))+"\n")

    outFile.close()
    
calculate_features("Data/EssentialsofML/trainingFiltered.csv","Data/EssentialsofML/trainingFilteredWithNewFeatures.csv")
calculate_features("Data/EssentialsofML/heldoutFiltered.csv","Data/EssentialsofML/heldoutFilteredWithNewFeatures.csv")


import numpy
import csv

reader=csv.reader(open("Data/EssentialsofML/trainingFilteredWithNewFeatures.csv","rb"),delimiter=',')
result=numpy.matrix(list(reader))[1:]

X = result[:,3:].astype('int')
y = numpy.squeeze(numpy.asarray(result[:,2]))
#X_new = SelectKBest(chi2, k=2000).fit_transform(X, y)


readerHeldout=csv.reader(open("Data/EssentialsofML/heldoutFilteredWithNewFeatures.csv","rb"),delimiter=',')
resultHeldout=numpy.matrix(list(readerHeldout))[1:]

heldoutX = resultHeldout[:,3:].astype('int')
heldouty = numpy.squeeze(numpy.asarray(resultHeldout[:,2]))
#heldoutX_new = SelectKBest(chi2, k=2000).fit_transform(heldoutX, heldouty)


"""
##cross validation for random forest classifier
#create model

clf3_easy = RandomForestClassifier(n_estimators=150,criterion='entropy',max_depth=60,min_samples_split=4,max_features='sqrt')
#clf3_balanced =  RandomForestClassifier(n_estimators=150,criterion='entropy',max_depth=60,min_samples_split=4,max_features='sqrt')
#clf3 = RandomForestClassifier(n_estimators=150,criterion='entropy',max_depth=60,min_samples_split=4,max_features='sqrt')

clf3_smote = RandomForestClassifier(n_estimators=150,criterion='entropy',max_depth=60,min_samples_split=4,max_features='sqrt')

#clf3_smote2 = RandomForestClassifier(n_estimators=150,criterion='entropy',max_depth=60,min_samples_split=4,max_features='sqrt')


ee=EasyEnsemble(ratio=0.8,replacement=True,n_subsets=1)
X_easy,Y_easy=ee.fit_sample(X,y)
print('Resampled dataset shape after easy {}'.format(Counter(Y_easy[0])))


smte = SMOTE(random_state=420,k_neighbors=55,m_neighbors=12,out_step=0.7)
X_smote,Y_smote = smte.fit_sample(X_easy,Y_easy)
print('Resampled dataset shape after smote after easy {}'.format(Counter(Y_smote)))

#print "X shape="
#print X.shape
#print "y shape="
#print y.shape


'''
print "X_easy shape="
print X_easy.shape
print "Y_easy shape="
print Y_easy.shape

print "X[0]_easy_shape="
print X_easy[0].shape
print "Y[0]_easy_shape="
print Y_easy[0].shape
'''


#X_smote2,Y_smote2 = smte.fit_sample(X_smote,Y_smote)
#print('Resampled dataset shape after smote2 {}'.format(Counter(Y_smote2)))
'''
increasing n_estimators increases accuracy as we are seeing more number of trees in the forest

min_samples_split or minimum number of samples required to split an internal node, when set to 1, leads to increased accuracy

max_features include number of features to consider when looking for the best split
'''

#Evaluate a score by cross-validation
#scores3 = cross_val_score(clf3, X, y,cv=10)
scores3_easy = cross_val_score(clf3_easy,X_easy[0],Y_easy[0],cv=10)
scores3_smote=cross_val_score(clf3_smote,X_smote,Y_smote,cv=10)
#scores3_smote2=cross_val_score(clf3_smote2,X_smote2,Y_smote2,cv=10)


#print "X_smote"
#print len(X_smote)
#report accuracy

#print("Accuracy for Random Forest Classifier: %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std() * 2))

#print("Accuracy for Random Forest Classifier using easy ensemble after smote: %0.2f (+/- %0.2f)" % (scores3_easy.mean(), scores3_easy.std() * 2))

print("Accuracy for Random Forest Classifier using SMOTE: %0.2f (+/- %0.2f)" % (scores3_smote.mean(), scores3_smote.std() * 2))

#print("Accuracy for Random Forest Classifier using SMOTE2 after SMOTE after easy: %0.2f (+/- %0.2f)" % (scores3_smote2.mean(), scores3_smote2.std() * 2))
"""



##random forest classifier on heldout test data
#create model
for i in range(6):
    j=2000-i*50
    skb = SelectKBest(chi2,k=j)
    skbobj = skb.fit(X,y)
    X_new = skbobj.transform(X)
    print X_new.shape
    heldoutX_new = skbobj.transform(heldoutX)
    print heldoutX_new.shape

    #X_new = X
    #heldoutX_new = heldoutX
    
    #clf = RandomForestClassifier(n_estimators=150,criterion='entropy',max_depth=60,min_samples_split=4,max_features='sqrt')
    
    #clf_easy = RandomForestClassifier(n_estimators=150,criterion='entropy',max_depth=60,min_samples_split=4,max_features='sqrt')
    dict ={'A':20,'B':1,'D':45,'T':1}
    clf_smote = RandomForestClassifier(n_estimators=150,criterion='entropy',max_depth=70,class_weight=dict,min_samples_split=4,max_features='sqrt')
    
    #clf_smote2 = RandomForestClassifier(n_estimators=150,criterion='entropy',max_depth=60,min_samples_split=4,max_features='sqrt')
    
    #clf_tree_smote=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=60,min_samples_split=4,presort=False)
    
    #clf_adaboost =AdaBoostClassifier (base_estimator=clf_tree_smote,n_estimators=145,learning_rate=1.0,random_state=420)
    
    #bc = BalanceCascade(ratio=0.1,random_state = 420,n_max_subset=10,classifier='random-forest')
    #X_bcas ,Y_bcas = bc.fit_sample(X,y)

    print('original dataset shape {}'.format(Counter(y)))
    
    #ee=EasyEnsemble(ratio=0.1,replacement=True,n_subsets=1)
    #X_easy,Y_easy=ee.fit_sample(X,y)
    #print('Resampled dataset shape after easy {}'.format(Counter(Y_easy[0])))


    #clcentroid=ClusterCentroids(ratio=0.09,random_state=420)
    #X_clc,Y_clc=clcentroid.fit_sample(X_new,y)
    #print('Resampled dataset shape after cluster centroids {}'.format(Counter(Y_clc)))

    #skb = SelectKBest(chi2,k=500)
    #skbobj = skb.fit(X_easy[0],Y_easy[0])
    #X_new = skbobj.transform(X_easy[0])
    #print X_new.shape
    #heldoutX_new = skbobj.transform(heldoutX)
    #print heldoutX_new.shape


    smte=SMOTE(ratio=0.7,random_state=420,k_neighbors=21,m_neighbors=6,out_step=0.7,kind='borderline1')
    X_smote,Y_smote=smte.fit_sample(X_new,y)

    print('Resampled dataset shape after smote{}'.format(Counter(Y_smote)))


    #ada=ADASYN(ratio=0.8,random_state=420)
    #X_ada,Y_ada = ada.fit_sample(X_new,y)
    #print('Resampled dataset shape after ADASYN{}'.format(Counter(Y_ada)))


    #print "X_bcas.shape="
    #print X_bcas.shape
    
    #print('Resampled dataset shape after easy {}'.format(Counter(Y_easy[0])))

    #print "x_easy[0].shape"
    #print X_easy[0].shape

    #print "x_smote.shape"
    #print X_smote.shape

    scores_smote = cross_val_score(clf_smote, X_smote, Y_smote,cv=10)
    print("Accuracy for Random Forest Classifier using SMOTE: %0.2f (+/- %0.2f)" % (scores_smote.mean(), scores_smote.std() * 2))


    #fit the model according to the training data.
    #clf.fit(X, y)
    #clf_easy.fit(X_easy[0],Y_easy[0])
    clf_smote.fit(X_smote,Y_smote)
    #clf_ada.fit(X_ada,Y_ada)
    #clf_smote2.fit(X_smote2,Y_smote2)
    #clf_tree_smote.fit(X_smote,Y_smote)
    #clf_adaboost.fit(X_smote,Y_smote)


    #predict labels for heldout data
    #predictedy = clf.predict(heldoutX)
    
    #easy_predictedy = clf_easy.predict(heldoutX)
    smote_predictedy = clf_smote.predict(heldoutX_new)
    #adaboost_predictedy=clf_adaboost.predict(heldoutX_new)
    
    #smote2_predictedy = clf_smote2.predict(heldoutX)
    '''
    #easy importance feature plot
    importances_easy = clf_easy.feature_importances_
    for f in range(0,2000):
    print("%d. feature %d (%f)" % (f + 1, f, importances_easy[f]))
    plt.figure()
    plt.title("Feature importances after easy")
    std=np.std([tree.feature_importances_ for tree in clf_easy.estimators_],axis=0)
    plt.bar(range(0,2000),importances_easy,color="r",yerr=std,align="center")
    plt.xticks(range(0,2000))
    plt.xlim([-1,X_easy[0].shape[1]])
    '''
    '''
    #smote feature plot
    importances_smote = clf_tree_smote.feature_importances_
    #indices = np.argsort(importances_smote)[::-1]

    for f in range(0,2000):
    print("%d. feature %d (%f)" % (f + 1, f, importances_smote[f]))
    plt.figure()
    plt.title("Feature importances after smote after easy")
    std=np.std([tree.feature_importances_ for tree in clf_tree_smote.estimators_],axis=0)
    plt.bar(range(0,2000),importances_smote,color="r",yerr=std,align="center")
    plt.xticks(range(0,2000))
    plt.xlim([-1,X_smote.shape[1]])
    '''

    #print precision recall table
    #print(classification_report(heldouty, predictedy))

    #report accuracy on heldout data
    #print "Accuracy= ",clf.score(heldoutX, heldouty)


    #report after applying easy ensemble
    #print (classification_report(heldouty , easy_predictedy))    
    #print "easy Accuracy= ",clf_easy.score(heldoutX,heldouty)

    #report after applying smote
    print (classification_report(heldouty , smote_predictedy))
    print "smote after easy Accuracy= ",clf_smote.score(heldoutX_new,heldouty)

    #report after applying smote and adaboost
    #print (classification_report(heldouty , adaboost_predictedy))
    #print "adaboost after smote Accuracy= ",clf_adaboost.score(heldoutX_new,heldouty)

    #report after applying ADASYN
    #print (classification_report(heldouty , ada_predictedy))
    #print "ADASYN Accuracy= ",clf_ada.score(heldoutX_new,heldouty)

    #plt.show()
