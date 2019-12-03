from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, request, jsonify, render_template, redirect
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import random

app = Flask(__name__)

from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

def evaluate_score(Y_test,predict):
    loss = hamming_loss(Y_test,predict)
    print("Hamming_loss : {}".format(loss*100))
    accuracy = accuracy_score(Y_test,predict)
    print("Accuracy : {}".format(accuracy*100))
    try :
        loss = log_loss(Y_test,predict)
    except :
        loss = log_loss(Y_test,predict.toarray())
    print("Log_loss : {}".format(loss))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Read the csv file into dataframe df
    df = pd.read_csv("train.csv")
    n = 159571  # number of records in file
    s = 25000  # desired sample size
    filename = "train.csv"
    skip = sorted(random.sample(range(n), n - s))
    df = pd.read_csv(filename, skiprows=skip)
    df.columns = ["id", "message", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    df = df.reindex(np.random.permutation(df.index))

    comment = df['message']
    comment = comment.as_matrix()

    label = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
    label = label.as_matrix()

    comments = []
    labels = []

    for ix in range(comment.shape[0]):
        if len(comment[ix]) <= 400:
            comments.append(comment[ix])
            labels.append(label[ix])

    labels = np.asarray(labels)
    import string
    print(string.punctuation)
    punctuation_edit = string.punctuation.replace('\'', '') + "0123456789"
    print(punctuation_edit)
    outtab = "                                         "
    trantab = str.maketrans(punctuation_edit, outtab)

    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')

    stop_words = stopwords.words("english")
    for x in range(ord('b'), ord('z') + 1):
        stop_words.append(chr(x))

    import nltk
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    # create objects for stemmer and lemmatizer
    lemmatiser = WordNetLemmatizer()
    stemmer = PorterStemmer()
    # download words from wordnet library
    nltk.download('wordnet')

    for i in range(len(comments)):
        comments[i] = comments[i].lower().translate(trantab)
        l = []
        for word in comments[i].split():
            l.append(stemmer.stem(lemmatiser.lemmatize(word, pos="v")))
        comments[i] = " ".join(l)

    # import required library
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    # create object supplying our custom stop words
    count_vector = TfidfVectorizer(stop_words=stop_words)
    # fitting it to converts comments into bag of words format
    tf = count_vector.fit_transform(comments)

    # print(count_vector.get_feature_names())
    print(tf.shape)

    def shuffle(matrix, target, test_proportion):
        ratio = int(matrix.shape[0] / test_proportion)
        X_train = matrix[ratio:, :]
        X_test = matrix[:ratio, :]
        Y_train = target[ratio:, :]
        Y_test = target[:ratio, :]
        return X_train, X_test, Y_train, Y_test

    X_train, X_test, Y_train, Y_test = shuffle(tf, labels, 3)

    from sklearn.naive_bayes import MultinomialNB

    # clf will be the list of the classifiers for all the 6 labels
    # each classifier is fit with the training data and corresponding classifier
    if request.method == 'GET':
        text = request.args.get('text')
    elif request.method == 'POST':
        data = json.loads(request.get_data().decode('utf-8'))
        message = data['text']
        model = data['model']
        print(model)
        data = [message]
        vect = count_vector.transform(data)

        if model == 'MultinomialNB':
            clf = []
            for ix in range(6):
                clf.append(MultinomialNB())
                clf[ix].fit(X_train, Y_train[:, ix])

            my_prediction = []
            for ix in range(6):
                my_prediction.append(clf[ix].predict(vect)[0])
            print(my_prediction)

        elif model =='XGBoost':

            from sklearn.multiclass import OneVsRestClassifier
            from xgboost import XGBClassifier
            from sklearn.preprocessing import MultiLabelBinarizer

            clf = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=4))

            clf.fit(X_train, Y_train)
            my_prediction = clf.predict(vect)[0]
            print(my_prediction.shape)
            print(my_prediction)#=(my_prediction[0,:].toarray())[0]

        #evaluate_score(Y_test, my_prediction)
        #for prediction in my_prediction:
        results = []
        result_dict = dict()
        result_dict['Toxic'] = str(my_prediction[0])
        result_dict['Severely Toxic'] = str(my_prediction[1])
        result_dict['Obscene'] = str(my_prediction[2])
        result_dict['Threat'] = str(my_prediction[3])
        result_dict['Insult'] = str(my_prediction[4])
        result_dict['Identity Hate'] = str(my_prediction[5])

        results.append(json.dumps(result_dict))

        return jsonify(results)




if __name__ == '__main__':
    app.run(debug=True)
