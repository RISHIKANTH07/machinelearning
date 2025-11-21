import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
data=pd.read_csv('cancerknn.csv')
df=pd.DataFrame(data)
col=["diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst",]
df=df.drop(columns=["id"])
df["diagnosis"]=(df['diagnosis']=="M").astype(int)
print(df.head())
train,vaild,test =np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
def scale_dataframe(dataframe,oversample):
    x=df[col[:-1]].values
    y=df[col[0]].values
    scaler=StandardScaler()
    x=scaler.fit_transform(x)
    if(oversample):
        ros=RandomOverSampler()
        x,y=ros.fit_resample(x,y)
    data1=np.hstack((x,np.reshape(y,(len(y),1)))) 
    data2=np.hstack((x,np.reshape(y,(len(y),1)))) 

    return data1,x,y

def model_train(df,train):
    train,x_train,y_train=scale_dataframe(df,oversample=True)
    vaild,x_vaild,y_vaild=scale_dataframe(df,oversample=True)
    test,x_test,y_test=scale_dataframe(df,oversample=True)
    print(train.shape)
    print(x_train.shape)
    print(y_train.shape)
    model=KNeighborsClassifier()
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    #print("Train Set Results:",y_pred)
    print(classification_report(y_test,y_pred))
    print("Accuracy:",accuracy_score(y_train,y_pred))

    return model
train,x_train,y_train=scale_dataframe(df,oversample=True)
vaild,x_vaild,y_vaild=scale_dataframe(df,oversample=True)
test,x_test,y_test=scale_dataframe(df,oversample=True)
def model_visualize(model,x_test,y_test):
    y_pred=model.predict(x_test)
    plt.scatter(range(len(y_test)),y_test,color='blue',label='Actual')
    plt.scatter(range(len(y_pred)),y_pred,color='red',label='Predicted')
    plt.title('KNN Classifier: Actual vs Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Diagnosis')
    plt.legend()
    plt.show()
model=model_train(df,train)
model_visualize(model,x_test,y_test)
print(model) 

