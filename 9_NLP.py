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

## kernel 2
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



## kernel 3
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





#############################kernel 4
import numpy
import csv

reader=csv.reader(open("Data/EssentialsofML/trainingFilteredWithNewFeatures.csv","rb"),delimiter=',')
result=numpy.matrix(list(reader))[1:]

X = result[:,3:].astype('int')
y = numpy.squeeze(numpy.asarray(result[:,2]))

readerHeldout=csv.reader(open("Data/EssentialsofML/heldoutFilteredWithNewFeatures.csv","rb"),delimiter=',')
resultHeldout=numpy.matrix(list(readerHeldout))[1:]

heldoutX = resultHeldout[:,3:].astype('int')
heldouty = numpy.squeeze(numpy.asarray(resultHeldout[:,2]))




########################## random forest classifier
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



################################################# heldout test random
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


###################################################
