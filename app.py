from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap  # For feature impact visualizations

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Ensure static/graphs directory exists
os.makedirs('static/graphs', exist_ok=True)

# Load the trained model
model = joblib.load('static/model.pkl')

# Load the dataset (used for generating graphs)
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')  # Replace with your dataset
X = df.drop('DEATH_EVENT', axis=1)  # Replace 'DEATH_EVENT' with the actual target column name

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Predict route has been called!")  # Debugging line
    try:
        # Collect user input from the form
        features = [
            float(request.form['age']),
            float(request.form['anaemia']),
            float(request.form['creatinine_phosphokinase']),
            float(request.form['diabetes']),
            float(request.form['ejection_fraction']),
            float(request.form['high_blood_pressure']),
            float(request.form['platelets']),
            float(request.form['serum_creatinine']),
            float(request.form['serum_sodium']),
            float(request.form['sex']),
            float(request.form['smoking']),
            float(request.form['time']),
        ]

        # Prepare input for prediction
        input_data = np.array([features])

        # Check shape of input data
        print("Input data shape:", input_data.shape)  # Debugging line

        prediction = model.predict(input_data)[0]

        # Display prediction result
        prediction_text = 'At Risk of Heart Failure' if prediction == 1 else 'Not at Risk'

        # Generate graphs for feature impact
        generate_graphs(input_data)

        return render_template('predict.html', prediction=prediction_text)

    except Exception as e:
        flash(f"An error occurred: {str(e)}")
        return redirect(url_for('home'))

def generate_graphs(input_data):
    """Generate visualizations for feature impact."""
    try:
        # Feature Importance Bar Plot
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        sns.barplot(x=importances, y=X.columns)
        plt.title('Feature Importance (Model-Wide)')
        plt.savefig('static/graphs/feature_importance.png')
        plt.close()

        # Target Distribution Pie Chart
        plt.figure(figsize=(6, 6))
        df['DEATH_EVENT'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Not at Risk', 'At Risk'])
        plt.title('Heart Failure Risk Distribution')
        plt.savefig('static/graphs/target_distribution.png')
        plt.close()

        # Heatmap of Correlations
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.savefig('static/graphs/heatmap.png')
        plt.close()

        # SHAP Summary Plot (Global Feature Impact)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Check shape of SHAP values
        print("SHAP values shape:", [sv.shape for sv in shap_values])  # Debugging line

        plt.figure()
        shap.summary_plot(shap_values[1], X, show=False)
        plt.title('SHAP Summary Plot (Global Feature Impact)')
        plt.savefig('static/graphs/shap_summary.png')
        plt.close()

        # SHAP Force Plot (Local Feature Impact)
        shap.initjs()
        force_plot = shap.force_plot(
            explainer.expected_value[1], explainer.shap_values(input_data)[1], input_data, feature_names=X.columns
        )
        shap.save_html('static/graphs/shap_force.html', force_plot)

    except Exception as e:
        print(f"Error in generate_graphs: {e}")  # Debugging line

if __name__ == '__main__':
    app.run(debug=True)
