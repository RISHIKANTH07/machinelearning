from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import sklearn.metrics as confusion_matrix
import sklearn.metrics as classification_report




data=pd.read_csv('svm.csv')
df=pd.DataFrame(data)
df=df.drop(columns=["Unnamed: 32"])
print("NaN count:\n", df.isna().sum())
x=df.drop(columns=["diagnosis", "id",])
y=df["diagnosis"]=(df['diagnosis']=="M").astype(int)
train,test,valid=np.split(df.sample(frac=1),[int(0.6*len(df)),int(0.8*len(df))])
print(f"train{train.shape},test{test.shape},valid{valid.shape}")
print("NaN count:\n", df.isna().sum())

def descalingdata(df,oversample):

    x=df.drop(columns=["diagnosis","id"])
    y=df["diagnosis"]
    print("X shape before scaling:", x.shape)
    scaler=StandardScaler()
    x=scaler.fit_transform(x)
    if(oversample):
        ros=RandomOverSampler()
        x,y=ros.fit_resample(x,y)
    return x,y
x_train,y_train=descalingdata(train,oversample=True)
x_test,y_test=descalingdata(test,oversample=False)
valid_x,valid_y=descalingdata(valid,oversample=False)
print("X_train shape:", x_train)
print("y_train shape:", y_train)
model=SVC()
model.fit(x_train,y_train)
model_pred=model.predict(x_test)
print("Test Set Results:",model_pred)
print("Accuracy:",model.score(x_test,y_test)) 
final_matrix=confusion_matrix.confusion_matrix(y_test,model_pred)
print("Confusion Matrix:\n",final_matrix)
print("Classification Report:\n",classification_report.classification_report(y_test,model_pred))   
def model_visualize(predicted):
    plt.bar(['true positive','false postive','true negative','false positive'],[final_matrix[0][0],final_matrix[0][1],final_matrix[1][0],final_matrix[1][1]],color=['green','red','yellow','blue'])
    plt.title('SVM model')
    plt.legend()
    plt.show()
model_visualize(final_matrix)
