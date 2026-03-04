from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

app = Flask(__name__)

diet_data = pd.read_csv("C:\\Users\\ravic\\OneDrive\\Desktop\\augmented_dataset_10000_rows.csv")

gender_encoder = LabelEncoder()
goal_encoder = LabelEncoder()
diet_encoder = LabelEncoder()

diet_data['Gender'] = gender_encoder.fit_transform(diet_data['Gender'])
diet_data['Goal'] = goal_encoder.fit_transform(diet_data['Goal'])
diet_data['Diet Recommendation'] = diet_encoder.fit_transform(diet_data['Diet Recommendation'])

X = diet_data[['Age', 'Gender', 'Height (cm)', 'Weight (kg)', 'Goal']]
y = diet_data['Diet Recommendation']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)


model = RandomForestClassifier(n_estimators=8, max_depth=2, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy*100:.2f}%")


def calculate_bmi(weight, height_cm):
    height_m = height_cm / 100
    return weight / (height_m ** 2)

def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 24.9:
        return "Normal weight"
    elif 25 <= bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"

@app.route('/')
def home():
    return render_template('ok1.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    age = int(request.form['age'])
    gender = request.form['gender']
    height_cm = float(request.form['height'])
    weight_kg = float(request.form['weight'])
    goal = request.form['goal']

    gender_encoded = gender_encoder.transform([gender.capitalize()])[0]
    goal_encoded = goal_encoder.transform([goal])[0]

    bmi = calculate_bmi(weight_kg, height_cm)
    bmi_cat = bmi_category(bmi)

    diet_pred = model.predict([[age, gender_encoded, height_cm, weight_kg, goal_encoded]])
    diet_recommendation = diet_encoder.inverse_transform(diet_pred)[0]

    row = diet_data[(diet_data['Age'] == age) & (diet_data['Gender'] == gender_encoded) & 
                    (diet_data['Height (cm)'] == height_cm) & (diet_data['Weight (kg)'] == weight_kg) & 
                    (diet_data['Goal'] == goal_encoded)]

    if not row.empty:
        food_recommendation = row['Food Recommendation'].values[0]
        exercise_recommendation = row['Exercise Recommendation'].values[0]
        hydration_recommendation = row['Hydration Recommendation'].values[0]
        sleep_recommendation = row['Sleep Recommendation'].values[0]
    else:
        similar_rows = diet_data[diet_data['Goal'] == goal_encoded]
        if similar_rows.empty:
            similar_rows = diet_data

        food_recommendation = similar_rows.sample(1)['Food Recommendation'].values[0]
        exercise_recommendation = similar_rows.sample(1)['Exercise Recommendation'].values[0]
        hydration_recommendation = similar_rows.sample(1)['Hydration Recommendation'].values[0]
        sleep_recommendation = similar_rows.sample(1)['Sleep Recommendation'].values[0]

    recommendations = {
        "BMI": bmi,
        "BMI Category": bmi_cat,
        "Diet Recommendation": diet_recommendation,
        "Food Recommendation": food_recommendation,
        "Exercise Recommendation": exercise_recommendation,
        "Hydration Recommendation": hydration_recommendation,
        "Sleep Recommendation": sleep_recommendation
    }

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
