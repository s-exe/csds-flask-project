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
        # This is for donut
        # circle = plt.Circle(xy=(0,0), radius=.75, facecolor='white')
        # plt.gca().add_artist(circle)

        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        
        img_b64 = base64.b64encode(img.getvalue()).decode('utf8')

        return img_b64


def getDescription(prediction, explanation, test_instance):

    if (prediction[0] >= prediction[1]):
        verdict = f'negative with a certainty of {str(round(prediction[0] * 100, 2))}%'
        implication = 'unlikely'
    elif (prediction[0] < prediction[1]):
        verdict = f'positive with a certainty of {str(round(prediction[1] * 100, 2))}%'
        implication = 'likely'


    head = f"Based on the provided information, the model's predicted outlook was {verdict}.\
             This prediction means that it was {implication} that the applicant is able to \
             complete the loan period without being over 90 days overdue on a repayment.<br>"
    

    pos = "These are the positively contributing features which were beneficial during\
           the consideration process:"

    neg = "These were the negatively contributing features which can be improved to potentially\
           yield a more positive prediction:"
    

    Fpos = ''
    Fneg = ''
    for feature in explanation:
        if feature[1] > 0:
            Fpos = Fpos + f'<br>{featureDescription(feature[0], test_instance[feature[0]], 1)}'
        else:
            Fneg = Fneg + f'<br>{featureDescription(feature[0], test_instance[feature[0]], 0)}'

    a = f'{head}<br>'
    b = f'{pos}{Fpos}<br>'
    c = f'<br>{neg}{Fneg}'

    return a, b, c


def featureDescription(feature, value, flag):

    if feature == 'ExternalRiskEstimate':
        if flag == 1:
            feature_desc = f'The high value of {feature}: {value}.'
        else:
            feature_desc = f'The low value of {feature}: {value}, which would benefit from being greater.'
    
    elif feature == 'MSinceOldestTradeOpen':
        if flag == 1:
            feature_desc = f'The long history of {feature}: {value}.'
        else:
            feature_desc = f'The short history of {feature}: {value}, which being longer would be more preferable.'
    
    elif feature == 'MSinceMostRecentTradeOpen':
        if flag == 1:
            feature_desc = f'Recent activity in {feature}: {value}.'
        else:
            feature_desc = f'Lack of recent activity in {feature}: {value}, where more recent activity is desired.'
    
    elif feature == 'AverageMInFile':
        if flag == 1:
            feature_desc = f'High average months in {feature}: {value}.'
        else:
            feature_desc = f'Low average months in {feature}: {value}, a higher average would provide a more reliable history.'
    
    elif feature == 'NumSatisfactoryTrades':
        if flag == 1:
            feature_desc = f'Numerous satisfactory trades in {feature}: {value}.'
        else:
            feature_desc = f'Few satisfactory trades in {feature}: {value}, which would benefit from more satisfactory trades.'
    
    elif feature == 'NumTrades60Ever2DerogPubRec':
        if flag == 1:
            feature_desc = f'Few trades with 60+ delinquencies or public records in {feature}: {value}.'
        else:
            feature_desc = f'More trades with 60+ delinquencies or public records in {feature}: {value} may indicate credit risk, therefore should be minimised.'
    
    elif feature == 'NumTrades90Ever2DerogPubRec':
        if flag == 1:
            feature_desc = f'Few trades with 90+ delinquencies or public records in {feature}: {value}.'
        else:
            feature_desc = f'More trades with 90+ delinquencies or public records in {feature}: {value} may indicate credit risk, therefore should be minimised.'
    
    elif feature == 'PercentTradesNeverDelq':
        if flag == 1:
            feature_desc = f'High percentage of trades with no delinquencies in {feature}: {value}.'
        else:
            feature_desc = f'Low percentage of trades with no delinquencies in {feature}: {value}, which would benefit from being more consistent.'
    
    elif feature == 'MSinceMostRecentDelq':
        if flag == 1:
            feature_desc = f'A long period of time since the last deliquency in {feature}: {value}.'
        else:
            feature_desc = f'A short period of time since the last deliquency in {feature}: {value}, which should be avoided.'
    
    elif feature == 'MaxDelq2PublicRecLast12M':
        if flag == 1:
            feature_desc = f'Lowest delinquency level in the last 12 months in {feature}: {value}.'
        else:
            feature_desc = f'Higher delinquency level in the last 12 months in {feature}: {value}, would best be avoided to demonstrate lesser risk.'
    
    elif feature == 'MaxDelqEver':
        if flag == 1:
            feature_desc = f'Lowest delinquency level ever in {feature}: {value}.'
        else:
            feature_desc = f'Higher delinquency level ever in {feature}: {value}, would best be avoided to demonstrate lesser risk.'
    
    elif feature == 'NumTotalTrades':
        if flag == 1:
            feature_desc = f'The total number of trades in {feature}: {value}.'
        else:
            feature_desc = f'The total number of trades in {feature}: {value}, where a greater number may show consistency.'
    
    elif feature == 'NumTradesOpeninLast12M':
        if flag == 1:
            feature_desc = f'Open trades in the last 12 months in {feature}: {value}.'
        else:
            feature_desc = f'No open trades in the last 12 months in {feature}: {value}, where a larger number would demonstrate more reliability.'
    
    elif feature == 'PercentInstallTrades':
        if flag == 1:
            feature_desc = f'The percentage of installment trades in {feature}: {value}.'
        else:
            feature_desc = f'The percentage of installment trades in {feature}: {value}, which may benefit from more instances.'
    
    elif feature == 'MSinceMostRecentInqexcl7days':
        if flag == 1:
            feature_desc = f'No recent inquiries (excluding last 7 days) in {feature}: {value}.'
        else:
            feature_desc = f'Recent inquiries (excluding last 7 days) have been performed in {feature}: {value}, which may lead to doubt about the applicant.'
    
    elif feature == 'NumInqLast6M':
        if flag == 1:
            feature_desc = f'Few inquiries in the last 6 months in {feature}: {value}.'
        else:
            feature_desc = f'Too many inquiries in the last 6 months in {feature}: {value}, which may indicate problematic activity.'
    
    elif feature == 'NumInqLast6Mexcl7days':
        if flag == 1:
            feature_desc = f'Few inquiries in the last 6 months (excluding last 7 days) in {feature}: {value}.'
        else:
            feature_desc = f'Numerous inquiries in the last 6 months (excluding last 7 days) in {feature}: {value}, which may indicate problematic activity.'
    
    elif feature == 'NetFractionRevolvingBurden':
        if flag == 1:
            feature_desc = f'Good ratio of revolving burden in {feature}: {value}'
        else:
            feature_desc = f'Poor ratio of revolving burden in {feature}: {value}, which may benefit from being greater.'
    
    elif feature == 'NetFractionInstallBurden':
        if flag == 1:
            feature_desc = f'Good installment burden ratio in {feature}: {value}'
        else:
            feature_desc = f'Poor installment burden ratio in {feature}: {value}, where a more reasonable ratio would be beneficial.'
    
    elif feature == 'NumRevolvingTradesWBalance':
        if flag == 1:
            feature_desc = f'Number of revolving trades with a balance in {feature}: {value}.'
        else:
            feature_desc = f'Number of revolving trades with a balance in {feature}: {value} that may indicate credit risk.'
    
    elif feature == 'NumInstallTradesWBalance':
        if flag == 1:
            feature_desc = f'Number of installment trades with a balance in {feature}: {value}.'
        else:
            feature_desc = f'Number of  installment trades with a balance in {feature}: {value} that may indicate credit risk.'
    
    elif feature == 'NumBank2NatlTradesWHighUtilization':
        if flag == 1:
            feature_desc = f'Low rate of high utilization of bank/national trades in {feature}: {value}.'
        else:
            feature_desc = f'High rate of high utilization of bank/national trades in {feature}: {value}, which should not be too high.'
    
    elif feature == 'PercentTradesWBalance':
        if flag == 1:
            feature_desc = f'The percentage of trades with a balance in {feature}: {value}.'
        else:
            feature_desc = f'The percentage of trades with a balance in {feature}: {value}, which may indicate credit risk.'
    
    else:
        feature_desc = f'No description available for {feature}'
    
    return feature_desc

    
'''
                            This is a sample data for initial page loading
'''
sampling_data = pd.read_csv('H_test.csv')

# Generate a random index
random_index = random.randint(0, len(sampling_data) - 1)
# Select the random test instance from H_val_X
random_test_instance = sampling_data.iloc[random_index]

sample_data = random_test_instance.to_dict()

# Placeholder values for index route (not utilised)
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
        for cat in sample_data:
            print(str(cat))
            val = request.form.get(str(cat))
            sample_data[cat] = int(val)

    sample = pd.Series(sample_data)
    sample_shaped = sample.values.reshape(1,23)

    prediction, model = predict(sample_shaped)
    explanation = run_lime(sample, model)

    top10 = get_top10(explanation)
    intro, positives, negatives = getDescription(explanation.predict_proba, top10, sample_data)
    bar_plot_img = plot_to_img('bar', explanation)
    pie_chart_img = plot_to_img('pie', explanation)

    return render_template('dashboard.html', 
                           val1=sample_data['NumTotalTrades'],
                           val2=sample_data['ExternalRiskEstimate'],
                           val3=sample_data['MSinceOldestTradeOpen'],
                           val4=sample_data['NumInqLast6M'],
                           val5=sample_data['AverageMInFile'],
                           val6=sample_data['MSinceMostRecentTradeOpen'],
                           val7=sample_data['NumTrades60Ever2DerogPubRec'],
                           val8=sample_data['NumTrades90Ever2DerogPubRec'],
                           val9=sample_data['PercentTradesNeverDelq'],
                           val10=sample_data['MSinceMostRecentDelq'],
                           val11=sample_data['MaxDelq2PublicRecLast12M'],
                           val12=sample_data['MaxDelqEver'],
                           val13=sample_data['NumTradesOpeninLast12M'],
                           val14=sample_data['PercentInstallTrades'],
                           val15=sample_data['MSinceMostRecentInqexcl7days'],
                           val16=sample_data['NumInqLast6Mexcl7days'],
                           val17=sample_data['NetFractionRevolvingBurden'],
                           val18=sample_data['NetFractionInstallBurden'],
                           val19=sample_data['NumRevolvingTradesWBalance'],
                           val20=sample_data['NumInstallTradesWBalance'],
                           val21=sample_data['NumBank2NatlTradesWHighUtilization'], 
                           val22=sample_data['PercentTradesWBalance'],
                           val23=sample_data['NumSatisfactoryTrades'],  
                           bar_plot_url=bar_plot_img, 
                           pie_chart_url=pie_chart_img,
                           intro = intro,
                           positives = positives,
                           negatives = negatives

                           )

if __name__ == '__main__':
    app.run(debug=True,  threaded=False)