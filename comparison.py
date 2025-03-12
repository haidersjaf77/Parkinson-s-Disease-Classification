chart = {
        'Metric':["Accuracy", "F1-Score", "Recall", "Precision"],
        'Logistic Regression':[accuracy_score(y_test, pred_LR), f1_score(y_test, pred_LR), recall_score(y_test, pred_LR), precision_score(y_test, pred_LR)],
        'SVM':[accuracy_score(y_test, pred_SVM), f1_score(y_test, pred_SVM), recall_score(y_test, pred_SVM), precision_score(y_test, pred_SVM)],
        'Naive Bayes':[accuracy_score(y_test, pred_NB), f1_score(y_test, pred_NB), recall_score(y_test, pred_NB), precision_score(y_test, pred_NB)],
        'Decision Trees':[accuracy_score(y_test, pred_DT), f1_score(y_test, pred_DT), recall_score(y_test, pred_DT), precision_score(y_test, pred_DT)],
        'Random Forest':[accuracy_score(y_test, pred_RF), f1_score(y_test, pred_RF), recall_score(y_test, pred_RF), precision_score(y_test, pred_RF)],        
}
chart = pd.DataFrame(chart)
chart