import numpy as np

from gensim.models import utils
from gensim.models import Doc2Vec

from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.externals import joblib
from sklearn.decomposition import PCA

import plot_svm as p
import matplotlib.pyplot as plt

import os

train_epoch = 5

d2v_model_path= './model.d2v'
svm_model_path= './model_svm.pkl'

### if d2v model doesn't exist, quit program
if (os.path.isfile(d2v_model_path)):
    print('Imported d2v model...')
    d2vmodel = Doc2Vec.load(d2v_model_path)
else:
    sys.exit(0)

### throwing d2v vectors into array
# what the fuck is this ridiculous shit

print('Stashing d2v data into array...')

all_count= 2225
bs_count = 510
en_count = 386
pol_count= 417
sp_count = 511
te_count = 401

trainlist = []
labellist = []

def append_data(count, label, a):
    for i in range(count):
        string = label + str(i)
        trainlist.append(d2vmodel.docvecs[string])
        labellist.append(a)

append_data(bs_count, "BUSINESS_", 1)
append_data(en_count, "ENTERTAINMENT_", 2)
append_data(pol_count, "POLITICS_", 3)
append_data(sp_count, "SPORTS_", 4) 
append_data(te_count, "TECHNOLOGY_", 5)

trainlist = np.array(trainlist)
labellist = np.array(labellist)

### SVM and CV
print("Training SVC...")
svc = svm.SVC(gamma=0.001, C=1.0)

svc.fit(trainlist, labellist)
scores = cross_val_score(svc, trainlist, labellist, cv=5)
print(scores)


### Plot SVM

# feature extraction
print("Extracting features for plot...")
pca = PCA(n_components=2).fit(trainlist)
pca_2d = pca.transform(trainlist)

svc_2d = svm.SVC(gamma=0.001, C=1.0)
svc_2d.fit(pca_2d,labellist)

models = { svc_2d }
titles = ('Classified Vectors')

X = pca_2d[:,:2]
y = labellist

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = p.make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    p.plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
