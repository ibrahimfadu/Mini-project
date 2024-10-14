import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load your data
data = pd.read_csv('data/comple_n1.csv')  # Ensure your CSV is in the same directory

# Data preprocessing
data.fillna(0, inplace=True)
data = pd.get_dummies(data, columns=['COLLEGE', 'BRANCH', 'CATEGORIES'])

# Prepare features and target
X = data[['Cutoff_2019', 'Cutoff_2020', 'Cutoff_2021', 'Cutoff_2022', 'Cutoff_2023']]
y = data['Cutoff_2024']

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, 'cutoff_model.pkl')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_cutoff = None
    if request.method == 'POST':
        college = request.form['college']
        branch = request.form['branch']
        category = request.form['category']
        
        # Create a DataFrame for the user input
        user_row = pd.Series({'COLLEGE_' + college: 1, 
                              'BRANCH_' + branch: 1, 
                              'CATEGORIES_' + category: 1})
        user_row = user_row.reindex(data.columns, fill_value=0).to_frame().T
        
        # Extract historical cutoff values
        user_input = user_row[['Cutoff_2019', 'Cutoff_2020', 'Cutoff_2021', 'Cutoff_2022', 'Cutoff_2023']]

        # Predict the cutoff for 2025
        predicted_cutoff = model.predict(user_input)[0]

    return render_template('index.html', predicted_cutoff=predicted_cutoff)

if __name__ == '__main__':
    app.run(debug=True)
