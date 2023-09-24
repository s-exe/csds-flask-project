from flask import Flask, render_template, request
import pickle
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

app = Flask(__name__)

# Sample top five important features (replace this with actual feature list)
top_5_features = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']

# Sample model prediction function (replace this with  actual model)
def predict(input_data):
    model = pickle.load(open('model_v1.pkl', 'rb'))
    prediction = model.predict(input_data)
    # Replace this with your model prediction logic
    # return {'prediction': 'Yes', 'confidence': 0.75}
    return prediction

def create_plot():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.plot(x, y)
    plt.title('Sine Wave')
    plt.xlabel('x')
    plt.ylabel('y')

def plot_to_img():
    create_plot()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    img_b64 = base64.b64encode(img.getvalue()).decode()

    return img_b64


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


@app.route('/plot')
def plot():

    img_b64 = plot_to_img()

    html = f'<img src="data:image/png;base64,{img_b64}" class="blog-image">'

    return render_template_string(html)