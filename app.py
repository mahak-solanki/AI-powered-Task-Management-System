from flask import Flask, request, render_template
import pickle
from datetime import datetime
import re
import matplotlib.pyplot as plt
import pandas as pd
import os
import io
import base64
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def preprocess(text):
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    tokens = text.split()
    return " ".join([stemmer.stem(word) for word in tokens if word not in stop_words])


def predict_category(text):
    clf = pickle.load(open("task_classifier.pkl", "rb"))
    tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    processed = preprocess(text)
    vector = tfidf.transform([processed])
    return clf.predict(vector)[0]


def predict_priority(text):
    processed = preprocess(text)
    length = len(processed)
    model = pickle.load(open("priority_model.pkl", "rb"))
    return {0: "Low", 1: "Medium", 2: "High"}[model.predict([[length]])[0]]

 


data = pd.read_csv(r"C:\Users\ritik\OneDrive\Documents\New_synthetic_dataset.csv")

all_users = data['assignee'].unique().tolist()
def assign_user(category, priority, deadline):
    if isinstance(deadline, str):
        deadline = datetime.strptime(deadline, "%Y-%m-%d")
    if priority == "High" or deadline < datetime.now():
        return all_users[hash(category) % len(all_users)]



@app.route("/", methods=["GET", "POST"])
def home():
    output = ""
    pie_chart = None  # Base64 image string

    if request.method == "POST":
        desc = request.form["description"]
        deadline = request.form["deadline"]

        category = predict_category(desc)
        priority = predict_priority(desc)
        user = assign_user(category, priority, deadline)
        output = f"<b>Predicted_Category:</b> {category}<br><b>Predicted_Priority:</b> {priority}<br><b>Assigned_user:</b> {user}"

        # Update dataset with new task (simulate)
        new_row = {"task": desc, "deadline": deadline, "assignee": user}
        global data
        data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

        # Generate workload pie chart
    
        user_counts = data['assignee'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(user_counts, labels=user_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        pie_chart = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close()

    return render_template("index.html", output=output, pie_chart=pie_chart)



if __name__ == "__main__":
    app.run(debug=True)





