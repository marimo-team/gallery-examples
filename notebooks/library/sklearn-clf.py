# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair==5.5.0",
#     "anthropic==0.69.0",
#     "great-tables==0.19.0",
#     "marimo",
#     "mohtml==0.1.11",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
#     "polars==1.34.0",
#     "pyarrow==21.0.0",
#     "scikit-learn==1.7.2",
#     "scipy==1.16.2",
#     "skrub==0.6.2",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(auto_download=["html"])

with app.setup:
    import marimo as mo
    import altair as alt
    import numpy as np
    import pandas as pd
    import polars as pl
    import scipy.stats as stats
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from skrub import TableVectorizer


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Scikit-Learn classification workbook

    This notebook let's you bootstrap a model for scikit-learn quickly. The main point here is to get started quickly and to be able to iterate from. It's likely that you will need to dive deeper for the best model for each specific use-case.

    You can start by uploading a .csv file for classification. Alternatively you may also use a simulated dataset as a demo.
    """)
    return


@app.cell
def _():
    # Create UI elements
    file_upload = mo.ui.file(
        label="Upload CSV file", kind="area", filetypes=[".csv"], multiple=False
    )

    # Display the UI elements together
    mo.vstack(
        [
            mo.md("## Data Upload and Configuration"),
            file_upload,
            mo.md("We will assume a default dataset if none is provided."),
        ]
    )
    return (file_upload,)


@app.cell
def _(file_upload):
    # Load data - either from upload or use default
    if file_upload.value:
        # Read uploaded CSV file
        df = pl.read_csv(file_upload.value[0].contents)
        data_source = "Uploaded file"
    else:
        # Create default dataset if no file uploaded
        from sklearn.datasets import make_classification

        # Generate a classification dataset
        _X, _y = make_classification(
            n_samples=1000,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=2,
            weights=[0.6, 0.4],
            flip_y=0.05,
            class_sep=0.8,
            random_state=42,
        )

        # Create DataFrame with meaningful column names
        feature_cols = {f"feature_{i + 1}": _X[:, i] for i in range(_X.shape[1])}
        df = pl.DataFrame({**feature_cols, "target": _y})
        data_source = "Default synthetic dataset (sklearn make_classification)"

    # Display data info
    mo.md(
        f"**Data source:** {data_source}  \n**Shape:** {df.shape[0]} rows × {df.shape[1]} columns"
    )
    return (df,)


@app.cell
def _():
    run_button = mo.ui.run_button(label="Run Analysis")
    return (run_button,)


@app.cell
def _(df):
    target_selector = mo.ui.dropdown(
        options=df.columns,
        label="🎯 Select Target Column (variable to predict)",
        value=df.columns[-1] if df.columns else None,  # Default to last column
        full_width=True,
    )
    return (target_selector,)


@app.cell
def _(cv_folds, cv_iters, run_button, target_selector):
    mo.vstack([mo.md("### ⚙️ Configuration"), target_selector, cv_folds, cv_iters, run_button])
    return


@app.cell
def _():
    # Add slider for cross-validation folds
    cv_folds = mo.ui.slider(
        start=3, stop=10, value=5, step=1, label="Number of cross-validation folds"
    )
    cv_iters = mo.ui.slider(
        start=1, stop=100, value=10, step=1, label="Number of random searches to perform"
    )
    return cv_folds, cv_iters


@app.cell
def _(df, run_button, target_selector):
    mo.stop(not run_button.value)
    # Prepare data for modeling
    # Separate features and target
    X = df.drop(target_selector.value)
    y = df[target_selector.value]

    # Convert to pandas for skrub compatibility
    X_pd = X.to_pandas()
    y_pd = y.to_pandas()
    return X_pd, y_pd


@app.cell
def _():
    # Define pipelines with TableVectorizer
    # TableVectorizer automatically handles mixed types (numeric, categorical, text)

    # Pipeline 1: Logistic Regression
    logistic_pipeline = Pipeline(
        [
            ("vectorizer", TableVectorizer()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )

    # Pipeline 2: Histogram Gradient Boosting
    hist_pipeline = Pipeline(
        [
            ("vectorizer", TableVectorizer()),
            ("classifier", HistGradientBoostingClassifier(random_state=42)),
        ]
    )

    # Define parameter grids for randomized search
    logistic_params = {
        "classifier__C": stats.loguniform(0.001, 10),
        "classifier__penalty": ["l2"],
    }

    hist_params = {
        "classifier__max_iter": stats.randint(50, 300),
        "classifier__max_depth": stats.randint(3, 15),
        "classifier__learning_rate": stats.uniform(0.01, 0.3),
        "classifier__min_samples_leaf": stats.randint(10, 50),
    }
    return hist_params, hist_pipeline, logistic_params, logistic_pipeline


@app.cell
def _(
    X_pd,
    cv_folds,
    cv_iters,
    hist_params,
    hist_pipeline,
    logistic_params,
    logistic_pipeline,
    y_pd,
):
    # Perform randomized search with cross-validation
    # Set up stratified k-fold
    skf = StratifiedKFold(n_splits=cv_folds.value, shuffle=True, random_state=42)

    # Define scoring metrics with weighted average for multiclass
    scoring = {
        "f1": "f1_weighted",
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted"
    }

    # Logistic Regression
    logistic_search = RandomizedSearchCV(
        logistic_pipeline,
        param_distributions=logistic_params,
        n_iter=cv_iters.value,
        cv=skf,
        scoring=scoring,
        refit="f1",
        random_state=42,
        n_jobs=-1,
        return_train_score=True,
        error_score="raise"
    )

    # Histogram Gradient Boosting
    hist_search = RandomizedSearchCV(
        hist_pipeline,
        param_distributions=hist_params,
        n_iter=cv_iters.value,
        cv=skf,
        scoring=scoring,
        refit="f1",
        random_state=42,
        n_jobs=-1,
        return_train_score=True,
    )

    # Fit models
    logistic_search.fit(X_pd, y_pd)
    hist_search.fit(X_pd, y_pd);
    return hist_search, logistic_search


@app.function
def get_cv_metrics(search_obj, model_name):
    """Extract cross-validation metrics for the best model"""
    best_idx = search_obj.best_index_
    cv_results = search_obj.cv_results_

    metrics = {
        "Model": model_name,
        "F1 Score": cv_results["mean_test_f1"][best_idx],
        "F1 Std": cv_results["std_test_f1"][best_idx],
        "Accuracy": cv_results["mean_test_accuracy"][best_idx],
        "Accuracy Std": cv_results["std_test_accuracy"][best_idx],
        "Precision": cv_results["mean_test_precision"][best_idx],
        "Precision Std": cv_results["std_test_precision"][best_idx],
        "Recall": cv_results["mean_test_recall"][best_idx],
        "Recall Std": cv_results["std_test_recall"][best_idx],
        "Train F1": cv_results["mean_train_f1"][best_idx],
        "Train Accuracy": cv_results["mean_train_accuracy"][best_idx],
    }
    return metrics


@app.cell
def _(hist_search, logistic_search):
    # Get metrics for both models
    logistic_metrics = get_cv_metrics(logistic_search, "Logistic Regression")
    hist_metrics = get_cv_metrics(hist_search, "Gradient Boosting")

    # Create comparison dataframe
    comparison_df = pl.DataFrame([logistic_metrics, hist_metrics])

    # Format the display
    mo.vstack(
        [
            mo.md("### Test Set Performance (Mean ± Std)"),
            pl.DataFrame(
                {
                    "Model": comparison_df["Model"],
                    "F1 Score": [
                        f"{comparison_df['F1 Score'][i]:.4f} ± {comparison_df['F1 Std'][i]:.4f}"
                        for i in range(len(comparison_df))
                    ],
                    "Accuracy": [
                        f"{comparison_df['Accuracy'][i]:.4f} ± {comparison_df['Accuracy Std'][i]:.4f}"
                        for i in range(len(comparison_df))
                    ],
                    "Precision": [
                        f"{comparison_df['Precision'][i]:.4f} ± {comparison_df['Precision Std'][i]:.4f}"
                        for i in range(len(comparison_df))
                    ],
                    "Recall": [
                        f"{comparison_df['Recall'][i]:.4f} ± {comparison_df['Recall Std'][i]:.4f}"
                        for i in range(len(comparison_df))
                    ],
                }
            ),
            mo.md("### Train vs Test Comparison"),
            pl.DataFrame(
                {
                    "Model": comparison_df["Model"],
                    "Train F1": comparison_df["Train F1"].round(4),
                    "Test F1": comparison_df["F1 Score"].round(4),
                    "Train Accuracy": comparison_df["Train Accuracy"].round(4),
                    "Test Accuracy": comparison_df["Accuracy"].round(4),
                }
            ),
        mo.md("Note that these results should be seen as a starting point for an ML workflow. You should always dive deeper to understand when/how a model might fail.")]
    )
    return


if __name__ == "__main__":
    app.run()
