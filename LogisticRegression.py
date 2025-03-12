model_LR = LogisticRegression(random_state = 0)
model_LR.fit(x_train, y_train)
pred_LR = model_LR.predict(x_test)
pred_LR

print(classification_report(y_test, pred_LR))

cm = confusion_matrix(y_test, pred_LR)
plt.figure(figsize = (6, 4))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues')
plt.title("Confusion Matrix of Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

y_pred_proba = model_LR.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="Area under curve = "+str(auc))
plt.legend(loc=4)
plt.show()