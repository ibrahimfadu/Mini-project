from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load data from CSV file
df = pd.read_csv('data/comple_n1.csv')

# Handle NaN values by filling with a default value (e.g., 0) or using SimpleImputer
imputer = SimpleImputer(strategy='mean')
df[['Cutoff_2019', 'Cutoff_2020', 'Cutoff_2021', 'Cutoff_2022', 'Cutoff_2023']] = imputer.fit_transform(
    df[['Cutoff_2019', 'Cutoff_2020', 'Cutoff_2021', 'Cutoff_2022', 'Cutoff_2023']]
)

# Prepare features and target
X = df[['COLLEGE', 'BRANCH', 'CATEGORIES', 'Cutoff_2019', 'Cutoff_2020', 'Cutoff_2021', 'Cutoff_2022', 'Cutoff_2023']]
y = df['Cutoff_2024']

# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['COLLEGE', 'BRANCH', 'CATEGORIES']),
    ],
    remainder='passthrough'  # Keep other columns (the cutoff columns)
)

# Create a pipeline that combines the preprocessing with the Ridge model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])

# Fit the model
model.fit(X, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        college = request.form['college']
        branch = request.form['branch']
        category = request.form['category']

        # Retrieve the past cutoffs using .loc
        past_cutoffs_row = df.loc[
            (df['COLLEGE'] == college) & 
            (df['BRANCH'] == branch) & 
            (df['CATEGORIES'] == category)
        ]

        if not past_cutoffs_row.empty:
            # Get past cutoffs as a list
            past_cutoffs = past_cutoffs_row.iloc[0][['Cutoff_2019', 'Cutoff_2020', 'Cutoff_2021', 'Cutoff_2022', 'Cutoff_2023','Cutoff_2024']].tolist()
        else:
            # Handle the case where there is no matching record
            past_cutoffs = [0, 0, 0, 0, 0,0]  # Default values

        # Prepare the input for prediction
        input_data = pd.DataFrame({
            'COLLEGE': [college],
            'BRANCH': [branch],
            'CATEGORIES': [category],
            'Cutoff_2019': [past_cutoffs[0]],
            'Cutoff_2020': [past_cutoffs[1]],
            'Cutoff_2021': [past_cutoffs[2]],
            'Cutoff_2022': [past_cutoffs[3]],
            'Cutoff_2023': [past_cutoffs[4]],
        })

        # Make prediction
        predicted_cutoff = model.predict(input_data)[0]

        # Format the prediction to an integer
        formatted_cutoff = int(round(predicted_cutoff))

        # Calculate differences from previous cutoffs
        differences = [cutoff - formatted_cutoff for cutoff in past_cutoffs]

        # Redirect to the result page with previous cutoffs, predicted cutoff, and differences
        return render_template('result.html', 
                               college=college, 
                               branch=branch, 
                               category=category, 
                               past_cutoffs=past_cutoffs, 
                               predicted_cutoff=formatted_cutoff, 
                               differences=differences)

    # Prepare dropdown lists for the form
    return render_template('index.html', colleges=df['COLLEGE'].unique(), branches=df['BRANCH'].unique(), categories=df['CATEGORIES'].unique())

if __name__ == '__main__':
    app.run(debug=True)
