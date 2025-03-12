model_NB = NaiveBayes()
model_NB.fit(x_train, y_train)
pred_NB = model_NB.predict(x_test)
print('Training Accuracy: ', model_NB.score(x_train, y_train))
print('Testing Accuracy: ', model_NB.score(x_test, y_test))

print(classification_report(y_test, pred_NB))

cm = confusion_matrix(y_test, pred_NB)
plt.figure(figsize = (6, 4))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues')
plt.title("Confusion Matrix of Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

y_pred_proba = model_NB.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="Area under curve = "+str(auc))
plt.legend(loc=4)
plt.show()