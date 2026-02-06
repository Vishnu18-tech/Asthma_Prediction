import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("synthetic_asthma_dataset.csv")
df.drop(columns=['Patient_ID','Family_History','Occupation_Type','Comorbidities','Asthma_Control_Level'], inplace=True)
df['Allergies'] = df['Allergies'].fillna('Unknown')
#print(df.info())

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score,roc_curve,confusion_matrix,f1_score
from sklearn.linear_model import LogisticRegression
X=df.drop(columns=['Has_Asthma'],axis=1)
y=df['Has_Asthma']
catcols=X.select_dtypes(include='object').columns
numcols=X.select_dtypes(include=['int64', 'float64']).columns
preprocessing=ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),numcols),
        ('cat',OneHotEncoder(),catcols)
    ],
)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=Pipeline(steps=[
    ('preprocessor',preprocessing),
    ('regressor',LogisticRegression(class_weight='balanced',max_iter=1000))
])
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
y_proba=model.predict_proba(X_test)[:,1]
print(f"Accuracy(LR) : {accuracy_score(y_test,y_pred)}")
print(f"F1 Score(LR) : {f1_score(y_test,y_pred)}")
print(f"roc_auc_score(LR) : {roc_auc_score(y_test,y_proba)}")
print(f"Classification_Report(LR) : \n{classification_report(y_test,y_pred)}")
print(f"Confusion_Matrix(LR) : \n{confusion_matrix(y_test,y_pred)}")


from sklearn.svm import SVC
model_svm=Pipeline(steps=[
    ('preprocessor',preprocessing),
    ('classifier',SVC(class_weight='balanced',kernel='rbf',C=1.0,probability=True))
])
model_svm.fit(X_train,y_train)
y_pred_svm=model_svm.predict(X_test)
y_proba_svm=model_svm.predict_proba(X_test)[:,1]
print(f"Accuracy(SVC) : {accuracy_score(y_test,y_pred_svm)}")
print(f"F1 Score(SVC) : {f1_score(y_test,y_pred_svm)}")
print(f"roc_auc_score(SVC) : {roc_auc_score(y_test,y_proba_svm)}")
print(f"Classification_Report(SVC) : \n{classification_report(y_test,y_pred_svm)}")
print(f"Confusion_Matrix(SVC) : \n{confusion_matrix(y_test,y_pred_svm)}")


rocauc=roc_auc_score(y_test,y_proba)
fpr,tpr,thresholds=roc_curve(y_test,y_proba)
plt.figure()
plt.plot(fpr,tpr,label='area=%0.2f'%rocauc)
plt.plot([0,1],[0,1],'r--')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC CURVE(LR)",fontweight='bold')
plt.show()
rocauc=roc_auc_score(y_test,y_proba_svm)
fpr,tpr,thresholds=roc_curve(y_test,y_proba_svm)
plt.figure()
plt.plot(fpr,tpr,label='area=%0.2f'%roc_auc_score(y_test,y_proba_svm))
plt.plot([0,1],[0,1],'r--')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC CURVE(SVC)",fontweight='bold')
plt.show()