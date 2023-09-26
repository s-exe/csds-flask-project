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
    file = open('lime_model', 'rb')
    explainer = dill.load(file)
    file.close()
    explanation = explainer.explain_instance(sample, model.predict_proba)

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
    fig, ax = plt.subplots(figsize=(6.5, 6))

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

    plt.figure(figsize=(6.5, 5))

    # Create a pie chart
    plt.pie(explanation.predict_proba, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', textprops={'fontsize': 8}, startangle=140)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')

    # Title
    plt.title('HELOC Approval Probabilities')


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
    
    elif type == 'pie':
        create_pie_chart(explanation)

        img = io.BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)

        img_b64 = base64.b64encode(img.getvalue()).decode('utf8')

        return img_b64

# Placeholder values for index route
sample_data = {'ExternalRiskEstimate':87, 
                   'MSinceOldestTradeOpen':155, 
                   'MSinceMostRecentTradeOpen':4, 
                   'AverageMInFile':70, 
                   'NumSatisfactoryTrades':35, 
                   'NumTrades60Ever2DerogPubRec':0, 
                   'NumTrades90Ever2DerogPubRec':0, 
                   'PercentTradesNeverDelq':100, 
                   'MSinceMostRecentDelq':-7, 
                   'MaxDelq2PublicRecLast12M':7, 
                   'MaxDelqEver':8, 
                   'NumTotalTrades':36, 
                   'NumTradesOpeninLast12M':1, 
                   'PercentInstallTrades':19, 
                   'MSinceMostRecentInqexcl7days':0, 
                   'NumInqLast6M':0, 
                   'NumInqLast6Mexcl7days':0, 
                   'NetFractionRevolvingBurden':2, 
                   'NetFractionInstallBurden':-8, 
                   'NumRevolvingTradesWBalance':2, 
                   'NumInstallTradesWBalance':1, 
                   'NumBank2NatlTradesWHighUtilization':0, 
                   'PercentTradesWBalance':20}


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # print(sample_data)
        for i in range(1, 11):
            label = request.form.get("label"+str(i))
            val = request.form.get("feature"+str(i))
            sample_data[label] = int(val)

    sample = pd.Series(sample_data)
    sample_shaped = sample.values.reshape(1,23)

    prediction, model = predict(sample_shaped)
    explanation = run_lime(sample, model)

    top10 = get_top10(explanation)
    bar_plot_img = plot_to_img('bar', explanation)
    pie_chart_img = plot_to_img('pie', explanation)

    return render_template('plot.html', 
                           feat1=top10[0][0],
                           feat2=top10[1][0], 
                           feat3=top10[2][0], 
                           feat4=top10[3][0], 
                           feat5=top10[4][0], 
                           feat6=top10[5][0], 
                           feat7=top10[6][0], 
                           feat8=top10[7][0], 
                           feat9=top10[8][0], 
                           feat10=top10[9][0],
                           val1=sample_data[top10[0][0]],
                           val2=sample_data[top10[1][0]],
                           val3=sample_data[top10[2][0]],
                           val4=sample_data[top10[3][0]],
                           val5=sample_data[top10[4][0]],
                           val6=sample_data[top10[5][0]],
                           val7=sample_data[top10[6][0]],
                           val8=sample_data[top10[7][0]],
                           val9=sample_data[top10[8][0]],
                           val10=sample_data[top10[9][0]], 
                           bar_plot_url=bar_plot_img, 
                           pie_chart_url=pie_chart_img
                           )


# Dashboard output route
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

    top10 = get_top10(explanation)
    bar_plot_img = plot_to_img('bar', explanation)
    pie_chart_img = plot_to_img('pie', explanation)

    return render_template('plot.html', 
                           feat1=top10[0][0], 
                           feat2=top10[1][0], 
                           feat3=top10[2][0], 
                           feat4=top10[3][0], 
                           feat5=top10[4][0], 
                           feat6=top10[5][0], 
                           feat7=top10[6][0], 
                           feat8=top10[7][0], 
                           feat9=top10[8][0], 
                           feat10=top10[9][0], 
                           bar_plot_url=bar_plot_img, 
                           pie_chart_url=pie_chart_img)



if __name__ == '__main__':
    app.run(debug=True)