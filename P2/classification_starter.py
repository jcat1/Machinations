## This file provides starter code for extracting features from the xml files and
## for doing some learning.
##
## The basic set-up: 
## ----------------
## main() will run code to extract features, learn, and make predictions.
## 
## extract_feats() is called by main(), and it will iterate through the 
## train/test directories and parse each xml file into an xml.etree.ElementTree, 
## which is a standard python object used to represent an xml file in memory.
## (More information about xml.etree.ElementTree objects can be found here:
## http://docs.python.org/2/library/xml.etree.elementtree.html
## and here: http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/)
## It will then use a series of "feature-functions" that you will write/modify
## in order to extract dictionaries of features from each ElementTree object.
## Finally, it will produce an N x D sparse design matrix containing the union
## of the features contained in the dictionaries produced by your "feature-functions."
## This matrix can then be plugged into your learning algorithm.
##
## The learning and prediction parts of main() are largely left to you, though
## it does contain code that randomly picks class-specific weights and predicts
## the class with the weights that give the highest score. If your prediction
## algorithm involves class-specific weights, you should, of course, learn 
## these class-specific weights in a more intelligent way.
##
## Feature-functions:
## --------------------
## "feature-functions" are functions that take an ElementTree object representing
## an xml file (which contains, among other things, the sequence of system calls a
## piece of potential malware has made), and returns a dictionary mapping feature names to 
## their respective numeric values. 
## For instance, a simple feature-function might map a system call history to the
## dictionary {'first_call-load_image': 1}. This is a boolean feature indicating
## whether the first system call made by the executable was 'load_image'. 
## Real-valued or count-based features can of course also be defined in this way. 
## Because this feature-function will be run over ElementTree objects for each 
## software execution history instance, we will have the (different)
## feature values of this feature for each history, and these values will make up 
## one of the columns in our final design matrix.
## Of course, multiple features can be defined within a single dictionary, and in
## the end all the dictionaries returned by feature functions (for a particular
## training example) will be unioned, so we can collect all the feature values 
## associated with that particular instance.
##
## Two example feature-functions, first_last_system_call_feats() and 
## system_call_count_feats(), are defined below.
## The first of these functions indicates what the first and last system-calls 
## made by an executable are, and the second records the total number of system
## calls made by an executable.
##
## What you need to do:
## --------------------
## 1. Write new feature-functions (or modify the example feature-functions) to
## extract useful features for this prediction task.
## 2. Implement an algorithm to learn from the design matrix produced, and to
## make predictions on unseen data. Naive code for these two steps is provided
## below, and marked by TODOs.
##
## Computational Caveat
## --------------------
## Because the biggest of any of the xml files is only around 35MB, the code below 
## will parse an entire xml file and store it in memory, compute features, and
## then get rid of it before parsing the next one. Storing the biggest of the files 
## in memory should require at most 200MB or so, which should be no problem for
## reasonably modern laptops. If this is too much, however, you can lower the
## memory requirement by using ElementTree.iterparse(), which does parsing in
## a streaming way. See http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/
## for an example. 

import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import grid_search
from sklearn.grid_search import ParameterGrid

import util


def extract_feats(ffs, direc="train", global_feat_dict=None):
    """
    arguments:
      ffs are a list of feature-functions.
      direc is a directory containing xml files (expected to be train or test).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.

    returns: 
      a sparse design matrix, a dict mapping features to column-numbers,
      a vector of target classes, and a list of system-call-history ids in order 
      of their rows in the design matrix.
      
      Note: the vector of target classes returned will contain the true indices of the
      target classes on the training data, but will contain only -1's on the test
      data
    """
    fds = [] # list of feature dicts
    classes = []
    ids = [] 
    for datafile in os.listdir(direc):
        # extract id and true class (if available) from filename
        id_str,clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(util.malware_classes.index(clazz))
        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)
        rowfd = {}
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        # accumulate features
        [rowfd.update(ff(tree)) for ff in ffs]
        fds.append(rowfd)
        
    X,feat_dict = make_design_mat(fds,global_feat_dict)
    return X, feat_dict, np.array(classes), ids


def make_design_mat(fds, global_feat_dict=None):
    """
    arguments:
      fds is a list of feature dicts (one for each row).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.
       
    returns: 
        a sparse NxD design matrix, where N == len(fds) and D is the number of
        the union of features defined in any of the fds 
    """
    if global_feat_dict is None:
        all_feats = set()
        [all_feats.update(fd.keys()) for fd in fds]
        feat_dict = dict([(feat, i) for i, feat in enumerate(sorted(all_feats))])
    else:
        feat_dict = global_feat_dict
        
    cols = []
    rows = []
    data = []        
    for i in xrange(len(fds)):
        temp_cols = []
        temp_data = []
        for feat,val in fds[i].iteritems():
            try:
                # update temp_cols iff update temp_data
                temp_cols.append(feat_dict[feat])
                temp_data.append(val)
            except KeyError as ex:
                if global_feat_dict is not None:
                    pass  # new feature in test data; nbd
                else:
                    raise ex

        # all fd's features in the same row
        k = len(temp_cols)
        cols.extend(temp_cols)
        data.extend(temp_data)
        rows.extend([i]*k)

    assert len(cols) == len(rows) and len(rows) == len(data)
   

    X = sparse.csr_matrix((np.array(data),
                   (np.array(rows), np.array(cols))),
                   shape=(len(fds), len(feat_dict)))
    return X, feat_dict
    

## Here are two example feature-functions. They each take an xml.etree.ElementTree object, 
# (i.e., the result of parsing an xml file) and returns a dictionary mapping 
# feature-names to numeric values.
## TODO: modify these functions, and/or add new ones.
def first_last_system_call_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'first_call-x' to 1 if x was the first system call
      made, and 'last_call-y' to 1 if y was the last system call made. 
      (in other words, it returns a dictionary indicating what the first and 
      last system calls made by an executable were.)
    """
    c = Counter()
    in_all_section = False
    first = True # is this the first system call
    last_call = None # keep track of last call we've seen
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            if first:
                c["first_call-"+el.tag] = 1
                first = False
            last_call = el.tag  # update last call seen
            
    # finally, mark last call seen
    c["last_call-"+last_call] = 1
    return c

def system_call_count_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'num_system_calls' to the number of system_calls
      made by an executable (summed over all processes)
    """
    c = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            if el.tag not in c:
                c[el.tag] = 0
            else:
                c[el.tag] += 1
    return c

def counts_per_call(tree):
    c = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            if 'el.tag' in c.keys():
                c['el.tag'] += 1
            else:
                c['el.tag'] = 1
    return c

def counts_per_twocall_window(tree):
    c = Counter()
    in_all_section = False
    first = True
    second = True
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            #make calls into windows of 2
            if first:
                firstelem_window = el.tag
                first = False
            elif second:
                secondelem_window = el.tag
                second = False
                name = firstelem_window+"-"+secondelem_window
                if name in c.keys():
                    c[name] += 1
                else:
                    c[name] = 1
            else:
                firstelem_window = secondelem_window
                secondelem_window = el.tag
                name = firstelem_window+"-"+secondelem_window
                if name in c.keys():
                    c[name] += 1
                else:
                    c[name] = 1
    return c

def virus_specific_words(tree):
    c = Counter()
    in_all_section = False
    c['NETAPI32.dll.NetUserGetInfo'] = 0
    c['urlmon.dll'] = 0
    c['urlmon.dll.URLDownloadToFile'] = 0
    c['HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft'] = 0
    c['HKEY_LOCAL_MACHINE\Software\Microsoft'] = 0
    c['HKEY_LOCAL_MACHINE\Keyboard'] = 0
    c['HKEY_LOCAL_MACHINE\SOFTWARE\Classes'] = 0
    c['get_host_by_name'] = 0
    c['MSVBVM60.DLL'] = 0
    c['C:\WINDOWS\system32\sdra64.exe'] = 0
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            #autorun
            if el.tag == 'vm_protect':
                if 'target' in el.attrib:
                    if el.get('target') == 'NETAPI32.dll.NetUserGetInfo':
                        c['NETAPI32.dll.NetUserGetInfo'] = +1
        
            #lipler classified as a downloader virus, so has a lot of "urlmon.dll" and "downloadtofile"
                    elif el.get('target')[:28] == "urlmon.dll.URLDownloadToFile":
                        c['urlmon.dll.URLDownloadToFile'] = +1
                    elif el.get('target')[:10] == 'urlmon.dll':
                        c['urlmon.dll'] = +1
                    
            #magania --> password stealing virus, so uses cryptography in software class.
            elif el.tag == 'open_key':
                if 'key' in el.attrib:
                    if el.get('key')[:37] == 'HKEY_LOCAL_MACHINE\Software\Microsoft':
                        c['HKEY_LOCAL_MACHINE\Software\Microsoft'] = +1
                    elif el.get('key')[:26] == 'HKEY_CURRENT_USER\Keyboard':
                        c['HKEY_LOCAL_MACHINE\Keyboard'] = +1
            elif el.tag == 'query_value':
                if 'key' in el.attrib:
                    if el.get('key')[:36] == 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft':
                        c['HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft'] = +1

            
            #swizzor https://www.microsoft.com/en-us/wdsi/threats/malware-encyclopedia-description?Name=TrojanDownloader%3AWin32%2FSwizzor
            if el.tag == 'query_value':
                if 'key' in el.attrib:
                    if el.get('key')[:35] == 'HKEY_LOCAL_MACHINE\SOFTWARE\Classes':
                        c['HKEY_LOCAL_MACHINE\SOFTWARE\Classes'] = +1
            elif el.tag == 'get_host_by_name':
                c['get_host_by_name'] = +1
                
            #VB virus if it has MSVBVM60.DLL it is a VB virus. http://resources.infosecinstitute.com/anatomy-of-a-vb-virus/#gref
            elif el.tag == 'load_dll':
                if el.get('filename') == 'C:\WINDOWS\system32\MSVBVM60.DLL':
                    c['MSVBVM60.DLL'] = 1
                    
            #zbot creates copy of itself https://www.symantec.com/security_response/writeup.jsp?docid=2010-011016-3514-99&tabid=2
            elif el.tag == 'delete_file':
                if el.get('srcfile') == "C:\WINDOWS\system32\sdra64.exe":
                    c['C:\WINDOWS\system32\sdra64.exe'] = +1
                    
    return c
    

## The following function does the feature extraction, learning, and prediction
def main():
    train_dir = "train"
    test_dir = "test"
    outputfile = "sample_predictions.csv"  # feel free to change this or take it as an argument
    
    # TODO put the names of the feature functions you've defined above in this list
    ffs = [first_last_system_call_feats, system_call_count_feats, counts_per_call, counts_per_twocall_window, virus_specific_words]
    
    # extract features
    print "extracting training features..."
    X_train,global_feat_dict,t_train,train_ids = extract_feats(ffs, train_dir)
    print "done extracting training features"
    print

    # split data into train and validation sets
    print "spliting data into train and validation..."
    np.random.seed(9001)
    msk = np.random.rand(X_train.shape[0]) < 0.85
    X_train, X_valid = X_train[msk], X_train[~msk]
    t_train, t_valid = t_train[msk], t_train[~msk]
    print "done splitting"
    print
    
    # TODO train here, and learn your classification parameters

    # kNN Model
    print "learning kNN model..."
    knn_parameters = {'n_neighbors': range(1,21) + [25, 50, 100, 150], 'weights': ['uniform','distance'], 'p':[1, 2]}
    knn = grid_search.GridSearchCV(KNeighborsClassifier(), knn_parameters)
    knn.fit(X_train, t_train)

    print "kNN Hyperparameters"
    print "  n neighbors:", knn.best_estimator_.n_neighbors
    print "  weights:", knn.best_estimator_.weights
    print "  Minkowski power:", knn.best_estimator_.p
    print "-----"
    print "train score:", knn.score(X_train,t_train)
    print "validations score:", knn.score(X_valid,t_valid)
    print "done learning knn"
    print

    # Random Forest Model
    print "learning random forest model..."
    rf_parameters = {'n_estimators': [50, 500, 750, 1000], 'max_features': ['sqrt', 'log2', None], 'max_depth': [4, 25, 50, None], 'class_weight': ['balanced', None]}
    rf = grid_search.GridSearchCV(RandomForestClassifier(), rf_parameters)
    rf.fit(X_train, t_train)

    print "RF Hyperparameters"
    print "  n trees:", rf.best_estimator_.n_estimators
    print "  max depth:", rf.best_estimator_.max_depth
    print "  max features:", rf.best_estimator_.max_features
    print "  class weight:", rf.best_estimator_.class_weight
    print "-----"
    print "train score:", rf.score(X_train,t_train)
    print "validation score:", rf.score(X_valid,t_valid)
    print "done learning random forest"
    print

    # Logistic Regression Model
    print "learning logistic regression model..."
    lr_parameters = {'cv': range(3,11), 'fit_intercept':[True, False], 'penalty': ['l2']}
    lr = grid_search.GridSearchCV(LogisticRegressionCV(), lr_parameters)
    lr.fit(X_train, t_train)

    print "LR Hyperparameters"
    print "  k CV folds:", lr.best_estimator_.cv
    print "  fit intercept:", lr.best_estimator_.fit_intercept
    print "  penalty:", lr.best_estimator_.penalty
    print "-----"
    print "train score:", lr.score(X_train,t_train)
    print "validation score:", lr.score(X_valid,t_valid)
    print "done learning logistic regression"
    print

    # SVM Model
    print 'learning svm model...'
    svm_parameters = {'C':[.25, .5, 1, 2, 5]}
    svm = grid_search.GridSearchCV(SVC(), svm_parameters)
    svm.fit(X_train, t_train)

    print 'SVM Hyperparameters'
    print '  C values:', svm.best_estimator_.C
    print '-----'
    print 'train score:', svm.score(X_train,t_train)
    print 'validation score:', svm.score(X_valid,t_valid)
    print 'done learning svm'
    print

    # NN Model
    print 'learning nn model...'
    nn_parameters = {'activation': ['identity','logistic','tanh','relu'], 'alpha': [.0001, .0001, .001, .01, .1, 1]}
    nn = grid_search.GridSearchCV(MLPClassifier(), nn_parameters)
    nn.fit(X_train, t_train)

    print 'NN Hyperparameters'
    print '  activation function:', nn.best_estimator_.activation
    print '  loss coefficient (alpha value):', nn.best_estimator_.alpha
    print '-----'
    print 'train score:', nn.score(X_train,t_train)
    print 'validation score:', nn.score(X_valid,t_valid)
    print 'done learning nn'
    print
    
    # get rid of training data and load test data
    del X_train
    del t_train
    del train_ids
    print "extracting test features..."
    X_test,_,t_ignore,test_ids = extract_feats(ffs, test_dir, global_feat_dict=global_feat_dict)
    print "done extracting test features"
    print
    
    # TODO make predictions on text data and write them out
    print "making predictions..."
    knn_preds = knn.predict(X_test)
    rf_preds = rf.predict(X_test)
    lr_preds = lr.predict(X_test)
    svm_preds = svm.predict(X_test)
    nn_preds = nn.predict(X_test)
    print "done making predictions"
    print
    
    print "writing predictions..."
    util.write_predictions(knn_preds, test_ids, "knn_predictions.csv")
    util.write_predictions(rf_preds, test_ids, "rf_predictions.csv")
    util.write_predictions(lr_preds, test_ids, "lr_predictions.csv")
    util.write_predictions(svm_preds, test_ids, "svm_predictions.csv")
    util.write_predictions(nn_preds, test_ids, "nn_predictions.csv")
    print "done!"

if __name__ == "__main__":
    main()
    
