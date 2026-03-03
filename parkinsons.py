import shap
import matplotlib.pyplot as plt
import os

@app.route('/predictparkinsons', methods=['POST'])
@login_required
def predictparkinsons():

    # ---------------- GET INPUT ---------------- #
    feature_names = list(request.form.keys())
    int_features = [float(x) for x in request.form.values()]
    processed_feature = np.array([int_features])

    # ---------------- PREDICTION ---------------- #
    prediction = parkinsons_predict.predict(processed_feature)
    probability = parkinsons_predict.predict_proba(processed_feature)[0][1]
    prediction_score = round(probability, 4)

    # ---------------- RISK CLASSIFICATION ---------------- #
    if probability < 0.30:
        risk = "Low"
        uncertainty = "High Confidence"
        advice = "Healthy vocal patterns detected. Maintain regular monitoring."
    elif probability < 0.60:
        risk = "Moderate"
        uncertainty = "Moderate Confidence"
        advice = "Some vocal irregularities observed. Clinical screening recommended."
    elif probability < 0.85:
        risk = "High"
        uncertainty = "Moderate Confidence"
        advice = "Neurological consultation and voice analysis strongly advised."
    else:
        risk = "Critical"
        uncertainty = "Lower Confidence (High Probability)"
        advice = "Immediate neurological evaluation required."

    # ---------------- SHAP EXPLAINABILITY ---------------- #
    explainer = shap.Explainer(parkinsons_predict)
    shap_values = explainer(processed_feature)

    # Get top 3 important features
    shap_abs = np.abs(shap_values.values[0])
    top_indices = np.argsort(shap_abs)[-3:]
    top_features = [feature_names[i] for i in top_indices]

    # ---------------- SAVE SHAP GRAPH ---------------- #
    plt.figure()
    shap.plots.bar(shap_values[0], show=False)
    graph_path = os.path.join("static", "shap_parkinsons.png")
    plt.savefig(graph_path, bbox_inches='tight')
    plt.close()

    # ---------------- SAVE TO DATABASE ---------------- #
    new_prediction = Prediction(
        username=current_user.username,
        disease_type="Parkinsons",
        result=f"{risk} Risk"
    )
    db.session.add(new_prediction)
    db.session.commit()

    return render_template(
        "parkinsons.html",
        prediction_score=prediction_score,
        risk=risk,
        uncertainty=uncertainty,
        top_features=top_features,
        advice=advice,
        shap_image="shap_parkinsons.png"
    )