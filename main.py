
from flask import Flask, render_template, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
from joblib import load
import sklearn
import json


pipeline = load("model_classification.joblib")

app = Flask(__name__)
CORS(app)
api = Api(app)


class status (Resource):
    def get(self):
        try:
            return {'RESPONSE': 'Welcome to Twitter Hate-Speech Restful Api',
            'MODEL' : 'logistic regression',
            'USAGE' : 'In the URL secction   Add the class function "/analyze/"  then join with any sentence you wish to analyze',
                    'E.G' : 'http://localhost:5000/analyze/hello my name is Alaran'}
        except:
            return {'data': 'An Error Occurred during fetching Api'}

class analyze(Resource):
    def get(self, text):
        res = pipeline.predict([text])
        proba= ((pipeline.predict_proba([text]))[0])[0] * 100
        percent = " %"
        if res == 1 and proba <= 20:
            return jsonify({"OUTPUT" : "Something is fishy but HATE SPEECH is far", "Prediction probability is " : proba})
        elif res == 1 and proba <=40:
            return jsonify({"OUTPUT" : "WORD or SENTENCE likely to have HATE SPEECH", "Prediction probability is " : proba})
        elif res == 1 and proba <= 60:
            return jsonify({"OUTPUT" : "SLIGHT HATE SPEECH DETECTED!", "Prediction probability is " : proba})
        elif res == 1 and proba <= 80:
            return jsonify({"OUTPUT" : "HATE SPEECH IS DETECTED", "Prediction probability is " : proba})
        elif res == 1 and proba <= 100:
            return jsonify({"OUTPUT" : "HATE SPEECH IS DETECTED DEFINITELY", "Prediction probability is " : proba})
        elif res == 0:
            return jsonify({"OUTPUT" : "No Hate speech is found", "Prediction probability is " : proba})
        else:
            return jsonify({"OUTPUT" : "Nothing is found"})

api.add_resource(status, '/')
api.add_resource(analyze, '/analyze/<string:text>')

if __name__ == '__main__':
    app.run(debug=True)
