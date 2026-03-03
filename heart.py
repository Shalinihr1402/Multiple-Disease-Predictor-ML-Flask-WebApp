import shap
import matplotlib.pyplot as plt
import os

@app.route('/predictheartdisease', methods=['POST'])
@login_required
def predictheartdisease():

    # ---------------- FEATURE NAMES ---------------- #
    feature_names = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal"
    ]

    # ---------------- GET INPUT ---------------- #
    int_features = [float(x) for x in request.form.values()]
    processed_feature = np.array([int_features])

    # ---------------- PREDICTION ---------------- #
    prediction = heart_predict.predict(processed_feature)
    probability = heart_predict.predict_proba(processed_feature)[0][1]
    prediction_score = round(probability, 4)

    # ---------------- RISK CLASSIFICATION ---------------- #
    if probability < 0.30:
        risk = "Low"
        uncertainty = "High Confidence"
        advice = "No significant cardiac abnormality detected. Maintain healthy lifestyle."
    elif probability < 0.60:
        risk = "Moderate"
        uncertainty = "Moderate Confidence"
        advice = "Early cardiac risk indicators present. Regular monitoring recommended."
    elif probability < 0.85:
        risk = "High"
        uncertainty = "Moderate Confidence"
        advice = "High likelihood of heart disease. Cardiology consultation advised."
    else:
        risk = "Critical"
        uncertainty = "Lower Confidence (Very High Probability)"
        advice = "Immediate cardiac evaluation recommended. Seek medical attention."

    # ---------------- SHAP EXPLAINABILITY ---------------- #
    explainer = shap.Explainer(heart_predict)
    shap_values = explainer(processed_feature)

    shap_abs = np.abs(shap_values.values[0])
    top_indices = np.argsort(shap_abs)[-3:]
    top_features = [feature_names[i] for i in top_indices]

    # ---------------- SAVE SHAP GRAPH ---------------- #
    plt.figure()
    shap.plots.bar(shap_values[0], show=False)
    graph_path = os.path.join("static", "shap_heart.png")
    plt.savefig(graph_path, bbox_inches='tight')
    plt.close()

    # ---------------- SAVE TO DATABASE ---------------- #
    new_prediction = Prediction(
        username=current_user.username,
        disease_type="Heart Disease",
        result=f"{risk} Risk"
    )
    db.session.add(new_prediction)
    db.session.commit()

    # ---------------- RETURN TEMPLATE ---------------- #
    return render_template(
        "heartdisease.html",
        prediction_score=prediction_score,
        risk=risk,
        uncertainty=uncertainty,
        top_features=top_features,
        advice=advice,
        shap_image="shap_heart.png"
    )