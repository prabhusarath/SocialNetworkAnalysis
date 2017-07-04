# coding: utf-8

"""
CS579: Assignment 2

In this assignment, you will build a text classifier to determine whether a
movie review is expressing positive or negative sentiment. The data come from
the website IMDB.com.

You'll write code to preprocess the data in different ways (creating different
features), then compare the cross-validation accuracy of each approach. Then,
you'll compute accuracy on a test set and do some analysis of the errors.

The main method takes about 40 seconds for me to run on my laptop. Places to
check for inefficiency include the vectorize function and the
eval_all_combinations function.

Complete the 14 methods below, indicated by TODO.

As usual, completing one method at a time, and debugging with doctests, should
help.
"""

# No imports allowed besides these.
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request


def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.

    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.

    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.

    >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'], 
          dtype='<U5')
    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'], 
          dtype='<U5')
    """
    if (keep_internal_punct):
        string_tokenize = re.sub(r"""'(?!(?!> '))|'(?!(?<! '))""",'',doc.lower())
        return (np.array(re.sub(r"""["?,$!@&#><^%]""",'', string_tokenize).split()))
    else:
        return (np.array(re.sub(r'\W',' ', doc.lower()).split()))  
    pass

def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    token_features_string,counter =(),0
    
    for strings in tokens:
        token_features_string += (str("token=") + str(strings),)
    
    for feats_words in list(token_features_string):
        feats[feats_words] += (counter+1)
    
    pass

def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.

    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)

    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
    windows_list,counter = (),0
    
    for win_range in range(len(tokens)-(k-1)):
        windows_list += (tokens[win_range:win_range+k],)  
    
    for tokens_in_windows in list(windows_list):
        for windows_size in range(k-1,k):
            for cartesian_prod in combinations(tokens_in_windows,windows_size):
                feats[str("token_pair=") + str(cartesian_prod[0])+"__"+str(cartesian_prod[1])] += (counter+1)
                
    pass

neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    tokens_items,negative_items,positive_items =(),(),()
    
    for tokenize_string in tokens:
        tokens_items += (tokenize_string.upper(),)
    
    
    for negative_string in neg_words:
        negative_items += (negative_string.upper(),)
        
        
    for positive_string in pos_words:
        positive_items += (positive_string.upper(),)
            
    total_positive_count= set(list(tokens_items)).intersection(list(positive_items))
    total_negative_count= set(list(tokens_items)).intersection(list(negative_items))
    feats['pos_words']=len(total_positive_count)
    feats['neg_words']=len(total_negative_count) 

    pass  

def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.

    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.

    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    feats = defaultdict(lambda: 0)
    
    for functions in feature_fns:
        functions(tokens, feats)
        
    return sorted(feats.items(), key=lambda x: x[0])
    pass

def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.

    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),

    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    total_doc_list,vocab_dicts,data,indices,indptr =[],{},[],[],[]
    
    for token_items in tokens_list:
        doc_list = featurize(token_items,feature_fns)
        for documents in doc_list:
            vocab_dicts[documents[0]]=documents[1]   
        total_doc_list.append(doc_list)

    if vocab == None:
        vocab={}
        for vocab_items in sorted(vocab_dicts.keys()):
            vocab[vocab_items]=len(vocab)

    values_dictionary ={}
    for docs_items in total_doc_list:
        for val_items in docs_items:
            if val_items[0] not in values_dictionary:
                values_dictionary[val_items[0]]=1
            else: 
                values_dictionary[val_items[0]]=values_dictionary[val_items[0]]+1
                        
    for doc_items in range(len(total_doc_list)):
        for dic_items in total_doc_list[doc_items]:
            if (values_dictionary[dic_items[0]] >= min_freq):
                if dic_items[0] in vocab:
                    data.append(dic_items[1])
                    indices.append(doc_items)
                    indptr.append(vocab[dic_items[0]])
    
    data_array = np.array(data,np.int64)
    indices_array = np.array(indices,np.int64)
    indptr_array = np.array(indptr,np.int64)
    
    return csr_matrix((data_array, (indices_array, indptr_array)), shape=(len(total_doc_list),len(vocab))),vocab  
    pass

def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)

def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).

    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.

    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    testing_accuracies = ()
    cross_validation_calculation = KFold(len(labels), k)
    for data_train, data_test in cross_validation_calculation:
        clf.fit(X[data_train], labels[data_train])
        new_values = clf.predict(X[data_test])
        accuracies_values = accuracy_score(labels[data_test], new_values)
        testing_accuracies += (accuracies_values,)
    average_testing_accuracy = np.mean(list(testing_accuracies))
    return average_testing_accuracy
    pass

def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.

    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.

    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).

    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])

    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.

      This list should be SORTED in descending order of accuracy.

      This function will take a bit longer to run (~20s for me).
    """
    all_combinations =[]
    for features in range(1,len(feature_fns)+1):
        all_combinations.append(combinations(feature_fns,features))
    combined_values=[]
    all_final_combinations=[[]]
    for list_vals in all_combinations:
        for vals in list_vals:
            combined_values.append(vals)
    all_data = [tuple(all_data_val) for all_data_val in (punct_vals,combined_values,min_freqs)]
    for all_data_val in all_data:
        all_final_combinations=[com+[val] for com in all_final_combinations for val in all_data_val]
    dicts_list=()
    for prod in all_final_combinations:
        comb_dicts={}
        prouct_values = tuple(prod)
        comb_dicts['punct']=prouct_values[0]
        comb_dicts['features']=prouct_values[1]
        comb_dicts['min_freq']=prouct_values[2]
        classifier = LogisticRegression()
        tokens_list = [tokenize(doc_list,prouct_values[0]) for doc_list in docs]
        csr_matrix,v= vectorize(tokens_list,list(prouct_values[1]),prouct_values[2])
        comb_dicts['accuracy'] = cross_validation_accuracy(classifier,csr_matrix,labels,5)
        dicts_list += (comb_dicts,)
    final_sorted_dicts=sorted(list(dicts_list), key=lambda k: k['accuracy'],reverse=True) 
    return final_sorted_dicts
    pass

def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
    accuracies_list=[]
    for accuracy in results:
        accuracies_list.append(accuracy['accuracy'])
    
    x = range(0,len(accuracies_list))
    y = sorted(accuracies_list)
    
    plt.plot(x, y)
    plt.xlabel('setting')
    plt.ylabel('accuracy')
    plt.savefig(r'accuracies.png')
    pass

def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting. For example, compute the mean accuracy of all runs with
    min_freq=2.

    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    mean_dict={}
    final_list =[]
    
    for values in results:
        for key,val in values.items():
            if key!='accuracy':
                if key=='features':
                    feature_funcs=[]
                    for func_names in val:
                        feature_funcs.append(func_names.__name__)
                    val=feature_funcs
                if key+str(val) in mean_dict.keys():
                    mean_dict[key+str(val)].append(values['accuracy'])
                else:
                    mean_dict[key+str(val)]=[values['accuracy']]
       
    for setting,acc in mean_dict.items():
        Addition_values=0
        for acc_vals in acc:
            Addition_values = Addition_values + acc_vals
        final_accuracy=float(Addition_values)/len(acc)
        final_list.append((final_accuracy,setting))

    return final_list
    pass

def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)

    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    classifier= LogisticRegression()
    tokenized_list= [tokenize(documents,best_result['punct']) for documents in docs]
    csr_matrix,vocab= vectorize(tokenized_list,list(best_result['features']),best_result['min_freq'])
    classifier.fit(csr_matrix,labels)
    return classifier,vocab
    pass

def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.

    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """
    positive_coefficients,negative_coefficients = (),()
    classifier_coefficients = clf.coef_[0]
    index_sort_val = np.argsort(classifier_coefficients)
    vocab_keys={}

    if label == 1:

        positive_values = index_sort_val[::-1][:n]
        for vals in positive_values:
            for keys_vocab,values_vocab in vocab.items():
                if vals==values_vocab:
                    vocab_keys[values_vocab]=keys_vocab

        for pos_vals in positive_values:
            positive_coefficients += ((vocab_keys[pos_vals],classifier_coefficients[pos_vals]),)
                     
        positive_sorted_list = sorted(list(positive_coefficients), key=lambda x: x[1], reverse=True)  
        return positive_sorted_list  

    else:

        negative_values = index_sort_val[:n]
        for vals in negative_values:
            for keys_vocab,values_vocab in vocab.items():
                if vals==values_vocab:
                    vocab_keys[values_vocab]=keys_vocab
        
        for neg_vals in negative_values:
            negative_coefficients += ((vocab_keys[neg_vals],abs(classifier_coefficients[neg_vals])),)
        
        negative_sorted_list = sorted(list(negative_coefficients), key=lambda x: x[1], reverse=True)   
        return negative_sorted_list 
    pass

def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.

    Note: use read_data function defined above to read the
    test data.

    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    test_docs, test_labels = read_data(os.path.join('data', 'test'))
    tokenized_list= [tokenize(documents,best_result['punct']) for documents in test_docs]
    csr_matrix,cols= vectorize(tokenized_list,list(best_result['features']),best_result['min_freq'],vocab)
    return test_docs,test_labels,csr_matrix
    pass

def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.

    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.

    Returns:
      Nothing; see Log.txt for example printed output.
    """
    predicted_values = clf.predict(X_test)
    predicted_probability = clf.predict_proba(X_test)
    resultant_list=()

    for documents in range(len(test_docs)):
        final_dicts={}
        if (predicted_values[documents] != test_labels[documents]):
            final_dicts['truth']=test_labels[documents]
            final_dicts['predicted']=predicted_values[documents]
            final_dicts['proba']=predicted_probability[documents]
            final_dicts['test_docs']=test_docs[documents]
            if(final_dicts['predicted']==0):
                final_dicts['proba'] = final_dicts['proba'][0]
            else:
                final_dicts['proba'] = final_dicts['proba'][1]
            resultant_list += (final_dicts,)
        
    misclassifed_docs=sorted(list(resultant_list), key=lambda k: k['proba'],reverse=True) 

    print("First Document")
    print(misclassifed_docs[0])
    print("Second Document")
    print(misclassifed_docs[1])
    print("Third Document")
    print(misclassifed_docs[2])
    print("Fourth Document")
    print(misclassifed_docs[3])
    print("Fifth Document")
    print(misclassifed_docs[4])
    pass

def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)


if __name__ == '__main__':
    main()
