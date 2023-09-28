from flask import Flask, render_template, request, render_template_string
import pickle
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import dill
import re
import lime
import random
from lime import lime_text
from lime.lime_text import LimeTextExplainer

app = Flask(__name__)



'''

            A function that loads in the trained RandomForestClassifier (currently v2) and predicts the user's loan approval probability

                Input:
                    - 2D array (1,23)

                Outputs:
                    - RandomForestClassifier model (object)
                    - model prediction (list) 

'''

def predict(sample_shaped):
    file = open('model_v2.pkl', 'rb')
    model = pickle.load(file)
    file.close()
    prediction = model.predict(sample_shaped)

    return prediction, model



'''

            A function that runs LIME on the RandomForestClassifier model and explains its prediction

                Inputs:
                    - pandas Series (23, 2)
                    - RandomForestClassifier model (object)

                Output:
                    - explanation of a prediction instance (object)

'''
def run_lime(sample, model):
    file = open('lime_model', 'rb')
    explainer = dill.load(file)
    file.close()
    explanation = explainer.explain_instance(sample, model.predict_proba)

    return explanation



'''

            A function that maps LIME explanations to features and returns the top 10 contributing features to the model prediction

                Input:
                    - explanation of a prediction instance (object)

                Output:
                    - mapped LIME output (list)
'''
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



''' 

            A function that generates a bar plot based on the top 10 contributing features to the model prediction generated by LIME

                Input:
                    - mapped LIME output (list)

'''
def create_bar_plot(lime_output_mapped):

    feature_names, values = zip(*lime_output_mapped)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6.5, 6))

    # Create horizontal bars for positive and negative values
    ax.barh(feature_names, [max(0, val) for val in values], color='#B3D3E8', label='Positive', alpha=0.7)
    ax.barh(feature_names, [min(0, val) for val in values], color='#769FCD', label='Negative', alpha=0.7)

    # Add labels and legend
    ax.set_xlabel('Feature Contribution')
    ax.set_title('LIME Feature Contributions')
    ax.legend()

    ax.xaxis.grid(True, linestyle='--', alpha=0.5)

    # Invert the y-axis to display the most important features at the top
    ax.invert_yaxis()

    # Show the plot
    plt.tight_layout()



''' 

            A function that generates a pie chart based on prediction probabilities for each categories generated by LIME

                Input:
                    - explanation of a prediction instance (object)

'''
# Generate pie chart for LIME output
def create_pie_chart(explanation):
    labels = ['Rejected (Negative)', 'Approved (Positive)']

    # Colors for the pie chart
    colors = ['#769FCD', '#B3D3E8']

    # Explode a slice to emphasize it
    explode = (0.1, 0)

    plt.figure(figsize=(6.5, 5))

    # Create a pie chart
    plt.pie(explanation.predict_proba, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', textprops={'fontsize': 8}, startangle=140)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')

    # Title
    plt.title('HELOC Approval Probabilities')




'''

            A function that converts generated plots to a png file to allow plots to be rendered in HTML

                Input:
                    - intended plot type (string)
                    - explanation of a prediction instance (object)

                Output:
                    - link to locally saved plot image (string)

'''

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




'''
                            This is a sample data for initial page loading
'''
sampling_data = pd.read_csv('H_test.csv')

# Generate a random index
random_index = random.randint(0, len(sampling_data) - 1)
# Select the random test instance from H_val_X
random_test_instance = sampling_data.iloc[random_index]

sample_data = random_test_instance.to_dict()

# Placeholder values for index route
'''sample_data = {'ExternalRiskEstimate':87, 
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
                   'PercentTradesWBalance':20}'''

'''
                            This is a sample data for initial page loading
'''




'''

explanation for app route below
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
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

    return render_template('dashboard.html', 
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

if __name__ == '__main__':
    app.run(debug=True)