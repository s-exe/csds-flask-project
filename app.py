import re
import os
import json
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
# from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, flash, redirect

app = Flask(__name__)
app.secret_key = 's3747783'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/categories')
def category():
    return render_template('categories.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/job-detail')
def job_detail():
    return render_template('job-detail.html')

@app.route('/job_list')
def job_list():
    return render_template('job_list.html')

@app.route('/accounting_finance')
def accounting_finance():
    return render_template('accounting_finance.html')

@app.route('/engineering')
def engineering():
    return render_template('engineering.html')

@app.route('/healthcare_nursing')
def health_nursing():
    return render_template('healthcare_nursing.html')

@app.route('/sales')
def sales():
    return render_template('sales.html')
    
@app.route('/<folder>/<filename>')
def listing(folder, filename):
    return render_template('/' + folder + '/' + filename + '.html')

@app.route('/post_job', methods=['GET', 'POST'])
def posting():
    if request.method == 'POST':
        title = request.form['title']
        desc = request.form['description']
        email = request.form['email']
        salary = request.form['salary']

        if request.form['button'] == "generate":
            ## Combining title and description as input for predicting category
            sentences = sent_tokenize(title + ' ' + desc)

            ## Initialising RegexpTokenizer
            regxpttrn = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
            tokenator = RegexpTokenizer(regxpttrn)
            
            ## Tokenizing words
            senttoken_list = []
            token_list = []

            for sentence in sentences:
                tokens = tokenator.tokenize(sentence)
                senttoken_list.append(tokens)

            for sent in senttoken_list:
                for token in sent:
                    lower_token = token.lower()
                    token_list.append(lower_token)
            
            ## Loading vocabulary from M1 Task 2
            f = open('vocab.json')
            vocab = json.load(f)

            ## Initialising CountVectorizer
            count_vectorizer = CountVectorizer(vocabulary = vocab)

            ## Processing tokens for generating count vectors
            tokens = []
            strings = ''
            flag = 1 # a flag variable that checks if it is the first iteration of the for loop

            for token in token_list:
                if flag == None:
                    strings = strings + ' ' + token
                
                else:
                    strings = strings + token 
                    flag = None
            
            tokens.append(strings)
            count_features = count_vectorizer.fit_transform(tokens)

            ## Loading and running classification model to predict category
            model = pickle.load(open('DecisionTree_model.sav', 'rb'))

            y_pred = model.predict(count_features)
            prediction = y_pred[0]

            ## Changing the outputs for some categories to proper formats
            if prediction == 'Healthcare_Nursing':
                prediction = 'Healthcare and Nursing'
            
            elif prediction == 'Accounting_Finance':
                prediction = 'Accounting and Finance'

            return render_template('post_job.html', prediction = prediction, title = title, description = desc, email = email, salary = salary)
        

        elif request.form['button'] == "post":
            category = request.form.get('categories')

            ## Flash message when user attempts to post a job without choosing a category
            if category == None:
                flash('Please choose a category', 'error')
                return render_template('post_job.html', title = title, description = desc)
            
            else:
                ## Initialise BeautifulSoup parser
                parser = BeautifulSoup(open('templates/post.html'), 'html.parser')

                ## Adding title
                title_pos = parser.find('h3', { 'class' : "mb-3" })
                title_pos.append(title)

                ## Adding description
                desc_pos = parser.find('h4', { 'class' : "mb-3"})
                desc_p = parser.new_tag('p')
                desc_p.append(desc)
                desc_pos.insert_after(desc_p)

                ## Adding salary (if entered)
                if salary != '':
                    salary_pos = parser.find('u', { 'name' : "salary"})
                    salary_pos.append(salary)
                     
                ## Adding email (if entered)
                if email != '':
                    email_pos = parser.find('p', { 'class' : "m-0"})
                    email_pos.append(email)


                ## Generating webindex for posting
                # change file path if you want to test
                paths = ['/Users/schmexyboi/Advanced Programming with Python /A2/M2/TEST/templates/accounting_finance',
                         '/Users/schmexyboi/Advanced Programming with Python /A2/M2/TEST/templates/engineering',
                         '/Users/schmexyboi/Advanced Programming with Python /A2/M2/TEST/templates/healthcare_nursing',
                         '/Users/schmexyboi/Advanced Programming with Python /A2/M2/TEST/templates/sales']
                high_idx = 0

                for path in paths:
                    files = os.listdir(path)

                    for file in files:
                        str_idx = str(file[:8])
                        int_idx = int(str_idx)
                        
                        if int_idx > high_idx:
                            high_idx = int_idx
                
                idx = high_idx + 1

                ## Generate html file for posting
                if category == 'Accounting and Finance':
                    category = 'accounting_finance'

                elif category == 'Healthcare and Nursing':
                    category = 'healthcare_nursing'

                filename = category + '/' + str(idx) + '.html'
                with open('templates/' + filename, 'w', encoding = 'utf-8') as file:
                    print(filename)
                    file.write(str(parser))

                return redirect('/' + filename.replace('.html', ''))
    
    else:
        return render_template('post_job.html')

