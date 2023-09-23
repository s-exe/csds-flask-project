from flask import Flask, render_template, request

app = Flask(__name__)

# Sample top five important features (replace this with actual feature list)
top_5_features = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']

# Sample model prediction function (replace this with  actual model)
def predict(input_data):
    # Replace this with your model prediction logic
    return {'prediction': 'Yes', 'confidence': 0.75}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = {}
        for feature in top_5_features:
            user_input[feature] = float(request.form.get(feature, 0.0))

        # Call machine learning model to make predictions
        prediction_result = predict(user_input)

        return render_template('dashboard.html', top_5_features=top_5_features, prediction_result=prediction_result)

    return render_template('dashboard.html', top_5_features=top_5_features)

if __name__ == '__main__':
    app.run(debug=True)
