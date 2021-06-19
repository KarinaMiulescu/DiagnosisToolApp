import pickle
from json import loads

import flask
import pandas
from flask import jsonify, request

app = flask.Flask(__name__)
app.config["DEBUG"] = True

loaded_model = pickle.load(open('.///current_model///model.pk', 'rb'))
loaded_columns_list: list = pickle.load(open('.///current_model///columns.pk', 'rb'))

print(f"Loaded model {loaded_model} ")
print(f"Loaded columns {loaded_columns_list} \n len: {len(loaded_columns_list)}")


@app.route('/', methods=['GET'])
def get_symptoms():
    body = {'symptoms': loaded_columns_list}
    response = jsonify(body)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/', methods=['POST'])
def post_symptoms():
    print(request.get_data())
    print(request.get_json())

    request_data = loads(str(request.get_data()).replace("b'", "").replace("'", ""))
    print(request_data)
    selected_symptoms: list = request_data['symptoms']
    x_predict = list()

    number_of_symptoms = 3
    symptoms_counter = 0
    for column in loaded_columns_list:
        if column in selected_symptoms:
            x_predict.append(1)
            symptoms_counter += 1
        else:
            x_predict.append(0)

    if symptoms_counter < number_of_symptoms:
        response = {'error': "Please select at least 3 known symptoms"}
        json = jsonify(response)
        json.status = 400
        return json

    df = pandas.DataFrame(columns=loaded_columns_list)
    df = df.append(pandas.Series(x_predict, index=df.columns), ignore_index=True)
    body = {'diagnostic': str(loaded_model.predict(df)[0])}
    response = jsonify(body)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


app.run()
