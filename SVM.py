model_SVM = svm.SVC() 
model_SVM.fit(x_train, y_train)
print('Traning Accuracy: ', model_SVM.score(x_train, y_train))
print('Testing Accuracy: ', model_SVM.score(x_test, y_test))

param_grid = {'kernel':['linear','rbf','poly'],'C': [0.5, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

grid_SVM = GridSearchCV(svm.SVC(), param_grid, scoring='f1', verbose = 3)
grid_SVM.fit(x_train, y_train)

print("Best Parameters: ", grid_SVM.best_params_)
print("Best Estimator", grid_SVM.best_estimator_)
pred_SVM = grid_SVM.predict(x_test)  
print("\n Classification Report: \n", classification_report(y_test, pred_SVM)) 

cm = confusion_matrix(y_test, pred_SVM)
plt.figure(figsize = (6, 4))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues')
plt.title("Confusion Matrix of SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

fpr, tpr, _ = metrics.roc_curve(y_test,  pred_SVM)
auc = metrics.roc_auc_score(y_test, pred_SVM)
plt.plot(fpr,tpr,label="Area under curve = "+str(auc))
plt.legend(loc=4)
plt.show()