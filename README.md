# pysembler

An automatic ensembler of machine learning models in Python.

Overview
- pysembler builds stacked/level-wise ensembles using scikit-learn–compatible models.
- It performs out-of-fold training to generate meta-features, reports validation scores per fold/model, and writes per-level predictions to CSV for both train and test.
- Supports classification (using predict_proba) and regression (using predict).

Features
- Multi-level stacking: define models per level; predictions from level N feed level N+1.
- K-fold cross-validation:
  - StratifiedKFold for classification
  - KFold for regression
- Custom optimization metric: pass any function f(y_true, y_pred) returning a scalar score.
- Automatic saving of per-level train/test predictions to CSV.

Runtime dependencies
- numpy
- pandas
- scikit-learn

Installation (development)
- Install dev tools:
```
pip install -r requirements_dev.txt
```
- Install the package in editable mode:
```
pip install -e .
```

Usage

Minimal example (binary classification with two base models and one meta model):

```
from sklearn import ensemble
from sklearn.metrics import roc_auc_score
import numpy as np
import tempfile

from pysembler import Ensembler

# Custom metric: expects y_true and y_pred (probabilities or scores)
def auc_on_positive_class(y_true, y_pred):
    # For binary classification, use probability of the positive class (column 1)
    return roc_auc_score(y_true, y_pred[:, 1])

# Define your ensemble as levels of models.
# Level 0 has two base models; Level 1 has a single meta model.
model_dict = {
    0: [
        ensemble.RandomForestClassifier(n_jobs=10, n_estimators=100),
        ensemble.ExtraTreesClassifier(n_jobs=10, n_estimators=100),
    ],
    1: [
        ensemble.GradientBoostingClassifier(n_estimators=100, max_depth=7),
    ],
}

# Example data
X = np.random.rand(1000, 100)
X_test = np.random.rand(100, 100)
y = np.random.randint(0, 2, 1000)

# Lengths must be provided to Ensembler
lentrain = X.shape[0]
lentest = X_test.shape[0]

# Level-0 training/test inputs must be provided per model in the same order as model_dict[0]
# For higher levels, pysembler uses generated predictions automatically.
train_data_dict = {0: [X, X]}       # one array per level-0 model
test_data_dict = {0: [X_test, X_test]}

with tempfile.TemporaryDirectory() as save_dir:
    ens = Ensembler(
        model_dict=model_dict,
        num_folds=5,
        task_type='classification',  # or 'regression'
        optimize=auc_on_positive_class,
        lower_is_better=False,
        save_path=save_dir,          # directory to write CSV predictions
    )

    # Fit on training data; writes train_predictions_level_*.csv to save_path
    ens.fit(train_data_dict, y, lentrain)

    # Predict on test data; writes test_predictions_level_*.csv to save_path
    test_level_predictions = ens.predict(test_data_dict, lentest)

    # Access in-memory predictions if needed
    # - Training meta-features: ens.train_prediction_dict[level] -> array of shape
    #   (n_samples, n_models_at_level * n_classes) for classification, or (n_samples, n_models_at_level) for regression.
    # - Test meta-features: ens.test_prediction_dict[level] -> same shape as above.
```

Key concepts and data shapes
- model_dict:
  - A dict mapping level index (int) to a list of scikit-learn–compatible estimators.
  - Example: {0: [ModelA(), ModelB()], 1: [MetaModel()]}.
- Training and test data:
  - For level 0 only, pass a dict with key 0 and value a list of inputs aligned with model_dict[0].
    - Each entry can be:
      - A 2D array (X) for standard tabular input, or
      - A list of arrays if the corresponding model expects multiple inputs.
  - For higher levels, pysembler uses its internally generated predictions; no user input required.
- Targets (y):
  - For classification, y is label-encoded internally (LabelEncoder).
  - For regression, y is used as provided.
- optimize:
  - Signature: optimize(y_true, y_pred) -> scalar score.
  - For classification, y_pred is the predict_proba matrix with shape (n_samples, n_classes).
  - For regression, y_pred is a 1D array of predictions.
  - Set lower_is_better accordingly (used for logging only).
- Folds:
  - num_folds determines the number of CV folds per level/model.

Outputs
- CSV files written to save_path:
  - Train predictions: train_predictions_level_<level>.csv
  - Test predictions: test_predictions_level_<level>.csv
- Shapes:
  - Classification: (n_samples, n_models_at_level * n_classes)
  - Regression: (n_samples, n_models_at_level)

Logging
- Verbose logging to stdout is enabled by default, including:
  - Fold-wise training and validation
  - Validation scores per model and fold
  - Mean and standard deviation of validation scores per model/level
  - File-saving events

Project structure
- pysembler/
  - __init__.py
  - ensembler.py
- tests/
  - test_pysembler.py
- setup.py
- setup.cfg
- requirements_dev.txt
- AUTHORS.rst
- README.md
- .editorconfig
- .gitignore

Development
- Code style: flake8 (configured via setup.cfg).
- Run tests:
```
pytest
```

License
- MIT

Authors
- Abhishek Thakur
- Eyad Sibai

Project URL
- https://github.com/abhishekkrthakur/pysembler
