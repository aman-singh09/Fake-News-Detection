from flask import Flask, render_template, jsonify, request
import json
import pickle

app = Flask(__name__, template_folder="../frontend/template")

def getNews(title):
    vectorization = pickle.load(open('../fakeNewsAlgo/vectorizer.sav', 'rb'))
    title = vectorization.transform([title])
    filename = '../fakeNewsAlgo/multinomialNB.sav'
    MNB = pickle.load(open(filename, 'rb'))
    filename = '../fakeNewsAlgo/logisticRegression.sav'
    LR = pickle.load(open(filename, 'rb'))
    filename = '../fakeNewsAlgo/knn.sav'
    KNN = pickle.load(open(filename, 'rb'))
    filename = '../fakeNewsAlgo/passiveAggressive.sav'
    PA = pickle.load(open(filename, 'rb'))
    filename = '../fakeNewsAlgo/DecisionTree.sav'
    DT = pickle.load(open(filename, 'rb'))

    LR_predict = LR.predict(title)[0]
    MNB_predict = MNB.predict(title)[0]
    KNN_predict = KNN.predict(title)[0]
    PA_predict = PA.predict(title)[0]
    DT_predict = DT.predict(title)[0]
    comparisonList = [MNB_predict, LR_predict, KNN_predict, PA_predict, DT_predict]
    true_freq = comparisonList.count(1)
    false_freq = comparisonList.count(0)
    if true_freq > false_freq:
        return "True"
    else:
        return "False"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api', methods=["POST"])
def base():
    print("Here")
    data = request.get_json()
    data = json.dumps(data)
    data_load = json.loads(data)
    newstitle = data_load['newsId']['description']
    print("in route")
    result = getNews(newstitle)
    print("getting result",result)
    solutions = {
        'title': newstitle,
        'Authenticity': result,
        'status': "Successfully Retrieved Result"
    }
    return jsonify(data=solutions)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
