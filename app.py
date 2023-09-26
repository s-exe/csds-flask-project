from flask import Flask, render_template, request, render_template_string
import pickle
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import dill
import re
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer

app = Flask(__name__)

# Load in RandomForestClassification, return sample prediction and model.
def predict(sample_shaped):
    file = open('model_v2.pkl', 'rb')
    model = pickle.load(file)
    file.close()
    prediction = model.predict(sample_shaped)

    return prediction, model

# Load in LIME model, return explanation of instance object
def run_lime(sample, model):
    print("INITIATED RUNNING LIME.....")
    file = open('lime_model', 'rb')
    explainer = dill.load(file)
    print("LIME MODEL LOADED.....")
    file.close()
    explanation = explainer.explain_instance(sample, model.predict_proba)
    print("EXPLANATION SUCCESSFULLY GENERATED")

    return explanation


# Gets top 10 features from LIME 
def get_top10(explanation):

    raw_data = explanation.as_list()

    feature_name_to_index = {
        'ExternalRiskEstimate': 0,
        'MSinceOldestTradeOpen': 1,
        'MSinceMostRecentTradeOpen': 2,
        'AverageMInFile': 3,
        'NumSatisfactoryTrades': 4,
        'NumTrades60Ever2DerogPubRec': 5,
        'NumTrades90Ever2DerogPubRec': 6,
        'PercentTradesNeverDelq': 7,
        'MSinceMostRecentDelq': 8,
        'MaxDelq2PublicRecLast12M': 9,
        'MaxDelqEver': 10,
        'NumTotalTrades': 11,
        'NumTradesOpeninLast12M': 12,
        'PercentInstallTrades': 13,
        'MSinceMostRecentInqexcl7days': 14,
        'NumInqLast6M': 15,
        'NumInqLast6Mexcl7days': 16,
        'NetFractionRevolvingBurden': 17,
        'NetFractionInstallBurden': 18,
        'NumRevolvingTradesWBalance': 19,
        'NumInstallTradesWBalance': 20,
        'NumBank2NatlTradesWHighUtilization': 21,
        'PercentTradesWBalance': 22
                            }
    
    lime_output_mapped = []

    for lime_rule in raw_data:
        feature_name = None
        coefficient = lime_rule[1]

        # Use regex to extract the feature index (a single digit)
        match = re.search(r'(^\d\d?| 0[1-9]?|[1-9]\d?) ', lime_rule[0])
        if match:
            feature_index = int(match.group())  # Extract the matched digit
            # Look up the feature name using the index
            for name, index in feature_name_to_index.items():
                if index == feature_index:
                    feature_name = name
                    break
        
        if feature_name:
            lime_output_mapped.append((feature_name, coefficient))
    
    return lime_output_mapped


# Generate bar plot from LIME output
def create_bar_plot(lime_output_mapped):

    feature_names, values = zip(*lime_output_mapped)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create horizontal bars for positive and negative values
    ax.barh(feature_names, [max(0, val) for val in values], color='green', label='Positive', alpha=0.7)
    ax.barh(feature_names, [min(0, val) for val in values], color='red', label='Negative', alpha=0.7)

    # Add labels and legend
    ax.set_xlabel('Feature Contribution')
    ax.set_title('LIME Feature Contributions')
    ax.legend()

    ax.xaxis.grid(True, linestyle='--', alpha=0.5)

    # Invert the y-axis to display the most important features at the top
    ax.invert_yaxis()

    # Show the plot
    plt.tight_layout()


# Generate pie chart for LIME output
def create_pie_chart(explanation):
    labels = ['Rejected (Negative)', 'Approved (Positive)']

    # Colors for the pie chart
    colors = ['red', 'green']

    # Explode a slice to emphasize it
    explode = (0.1, 0)

    plt.figure(figsize=(6, 6))

    # Create a pie chart
    plt.pie(explanation.predict_proba, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', textprops={'fontsize': 12}, startangle=140)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')

    # Title
    plt.title('HELOC Approval Probabilities')

    # Show the pie chart
    plt.show()



# Convert generated plot to img
def plot_to_img(type, explanation):

    if type == 'bar':
        lime_output_mapped = get_top10(explanation)
        create_bar_plot(lime_output_mapped)

        img = io.BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)

        img_b64 = base64.b64encode(img.getvalue()).decode('utf8')

        return img_b64
    
    else:
        create_pie_chart(explanation)

        img = io.BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)

        img_b64 = base64.b64encode(img.getvalue()).decode('utf8')

        return img_b64

top_5_features = [1, 2, 3, 4, 5]


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

@app.route('/plot')
def plot():
    sample_data = [87, 155, 4, 70, 35, 0, 0, 100, -7, 7, 8, 36, 1, 19, 0, 0, 0, 2, -8, 2, 1, 0, 20]
    sample_index = ['ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen', 'AverageMInFile', 
                'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec',
                'PercentTradesNeverDelq', 'MSinceMostRecentDelq', 'MaxDelq2PublicRecLast12M','MaxDelqEver', 
                'NumTotalTrades', 'NumTradesOpeninLast12M', 'PercentInstallTrades', 'MSinceMostRecentInqexcl7days',
                'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden',
                'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization',
                'PercentTradesWBalance']

    sample = pd.Series(sample_data, index = sample_index)
    sample_shaped = sample.values.reshape(1,23)

    prediction, model = predict(sample_shaped)
    explanation = run_lime(sample, model)

    bar_plot_img = plot_to_img('bar', explanation)
    pie_chart_img = plot_to_img('pie', explanation)
    return render_template('plot.html', bar_plot_url=bar_plot_img, pie_chart_url=pie_chart_img)


if __name__ == '__main__':
    app.run(debug=True)