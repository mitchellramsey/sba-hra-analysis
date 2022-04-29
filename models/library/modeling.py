import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (classification_report, accuracy_score, f1_score, 
                             matthews_corrcoef, confusion_matrix, roc_curve,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV

def createModel(model, X_train, y_train, X_test):
  """
   Takes in model and training data, and X_test data. Runs model and returns y_pred.
  """
  model.fit(X_train, y_train)
  return model.predict(X_test)

def createClassificationMetrics(y_pred, y_test, targetNames = ['paid', 'default']):
  """
  Takes y_predictions and y_test data. Returns accuracy score, classification report, f1 score, and matthew's Correlation
  Coefficient. Target names is an optional parameter that is provided default values for the SBA Loan Analysis.
  """
  acc_score = accuracy_score(y_pred, y_test)
  cr = classification_report(y_pred, y_test, target_names=targetNames)
  f1 = f1_score(y_pred, y_test)
  mcc = matthews_corrcoef(y_pred, y_test)
  return {
    'acc': acc_score,
    'f1': f1,
    'cr': cr,
    'mcc': mcc
  }

def runGridSearchAnalysis(model, param_grid, x_train, y_train, x_test, cv=5, scoring='f1'):
  """
    Runs a Grid Search Analysis on a provided model and param grid. Also requires x_train, y_train, and x_test datasets.
    Optional parameters are for cross validation (cv) defaulted at 5, and scoring which
    is defaulted at f1 for optimal study on the SBA Loan Analysis. Returns best_params and y_pred
  """
  search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)
  search.fit(x_train, y_train)
  y_pred = search.predict(x_test)
  return search.best_params_, y_pred

def createConfusionMatrix(y_test, y_pred, mod_info):
  """
    Creates and displays Confusion Matrix for the SBA Loan Analysis. Will return matrix labeled as 'True Paid', 'False Default', 
    'False Paid', and 'True Default'; along with the appropriate frequencies of each occurring. Returns the critical value
    for analysis: 'False Paid'
  """
  matrix = confusion_matrix(y_test, y_pred)
  group_names = ['True Paid', 'False Default', 'False Paid', 'True Default']
  labels = [f'{v1}\n{v2}' for v1,v2 in zip(group_names, matrix.flatten())]
  labels = np.asarray(labels).reshape(2,2)
  sns.heatmap(matrix, annot=labels, fmt='', cmap='Blues')
  plt.xlabel('Predicted Score')
  plt.ylabel('Actual Score')
  plt.title('{} Confusion Matrix using {} and {} Scaler'.format(mod_info['model'], mod_info['method'], mod_info['scaler']))
  plt.show()
  return matrix

def createFeatureImportanceChart(model, params, labels, x_train, y_train, rs=42):
  """
    Creates and displays feature importance chart for a model. Requires uninitialized model, param object, labels for graph,x_train, and y_train datasets. Optional random state variable may also be passed into function. 42 is default. Function does not return anything.
  """
  mod = model(**params, random_state=rs)
  mod.fit(x_train, y_train)

  feature_importance = mod.feature_importances_
  feature_importance = 100.0 * (feature_importance / feature_importance.max())
  sorted_idx = np.argsort(feature_importance)

  pos = np.arange(sorted_idx.shape[0]) + .5
  plt.figure(figsize=(10,10))
  plt.barh(pos, feature_importance[sorted_idx], align='center')
  plt.yticks(pos, labels[sorted_idx])
  plt.xlabel('Relative Importance')
  plt.title('Variable Importance')
  plt.show()

def appendModelingResults(mod_results, params_obj, mod_info, best_params, matrix, metrics):
    details = {
      'Model': mod_info['model'],
      'Method': mod_info['method'],
      'Scaler': mod_info['scaler'],
      'True Paid': matrix[0,0],
      'False Default': matrix[0,1],
      'False Paid': matrix[1,0],
      'True Default': matrix[1,1],
      'Accuracy': metrics['acc'],
      'AUC Score': metrics['auc'],
      'MCC': metrics['mcc'],
      'F1 Score': metrics['f1']
    }
    mod_results.append(details)
    params_obj[mod_info['model']][mod_info['method']][mod_info['scaler']] = best_params
    return mod_results, params_obj

def drawRocCurve(model, X_train, X_test, y_train, y_test, mod_info):
    """
    
    """
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    plt.plot(fpr,tpr)
    plt.plot([0,1], [0,1], 'k--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('{} ROC Curve using {} and {} Scaler'.format(mod_info['model'], mod_info['method'], mod_info['scaler']))

    return auc
