# Classification Models + Streamlit Deployment

## a. Problem statement
Build and compare five classification models (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) that predict whether a breast tumour is **malignant or benign** from measurements of cell nuclei taken from a digitised fine-needle-aspirate image. The models are evaluated with Accuracy, AUC, Precision, Recall, F1 and MCC, and are demonstrated in an interactive Streamlit app deployed on Streamlit Community Cloud.

## b. Dataset description
- **Source:** Breast Cancer Wisconsin (Diagnostic) dataset, UCI Machine Learning Repository (also distributed with scikit-learn as `load_breast_cancer`).
- **Instances:** 569
- **Features:** 30 numeric features: mean, standard error and "worst" value of 10 nucleus measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension).
- **Target:** `target`, binary: 0 = malignant, 1 = benign.
- **Split:** 80/20 stratified train/test split (`random_state=42`); test set = 114 rows, saved as `test_data.csv`.
- **Preprocessing:** `StandardScaler` inside a scikit-learn `Pipeline` with each classifier, so scaling is saved with each model.

## c. GitHub repository link
https://github.com/lakshman-battini/bc-diagnosis-classifier-app

Live app: https://bc-diagnosis-classifier.streamlit.app/

```
bc-diagnosis-classifier-app/
├── app.py               # Streamlit app
├── train_models.py      # trains the 5 models, saves models + test data + metrics
├── requirements.txt
├── README.md
├── test_data.csv        # held out test data (features + target)
├── full_dataset.csv
└── model/               # saved pipelines (.joblib) + metrics.csv
```

## d. Models used

### Comparison table (test set, 114 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

Precision, Recall and F1 are computed for the positive class (1 = benign).

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall (joint top accuracy 98.25%, highest AUC 0.9954, MCC 0.962). The classes are close to linearly separable once the features are standardised, so a simple linear model is enough. It is also fast and interpretable. |
| Decision Tree | Weakest model: lowest accuracy (92.1%), AUC (0.916) and MCC (0.834). A single depth-5 tree is high variance and overfits the training split; its hard leaf probabilities also give a poor ranking, hence the low AUC. An ensemble would likely help. |
| KNN | Very strong (accuracy 97.4%, AUC 0.988) and the only model with perfect recall (1.0), so no benign case was missed. Precision is lower (0.96), meaning it made a few false positives. Its performance depends on feature scaling, which the pipeline handles. |
| Naive Bayes | Good ranking ability (AUC 0.987) but lower accuracy (93.0%) and MCC (0.849). The independence assumption is violated, as many of the 30 features are strongly correlated (e.g. radius, perimeter, area), which hurts its decisions despite well-ordered scores. |
| Support Vector Machine | Ties Logistic Regression on accuracy, precision, recall, F1 and MCC, with essentially the same AUC (0.995). The RBF kernel captures any non-linearity, but the data does not need much, so it gives no gain over the linear model here. |
| **Overall winner for this dataset** | **Logistic Regression**, with SVM effectively tied. Logistic Regression is chosen because it matches SVM on every threshold metric, has the marginally higher AUC, and is simpler and more interpretable. |

Note: with only 114 test rows, one prediction changes accuracy by about 0.9 points, so differences between the top three models are within noise.

## How to run locally
```bash
pip install -r requirements.txt
python train_models.py        # regenerates models, test_data.csv, metrics
streamlit run app.py
```

## Streamlit app features
1. CSV upload of test data (must contain all 30 features + `target`)
2. Model selection dropdown (5 models)
3. Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
4. Confusion matrix and classification report
5. Side-by-side comparison of all models on the uploaded data

The app loads the saved models and only evaluates; it never retrains.
