import pandas as pd 
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler   
import matplotlib.pyplot as plt     
data=pd.read_csv('diabetes.csv')
df=pd.DataFrame(data)
y=df['Outcome'].values
x=df.drop('Outcome',axis=1).values
#print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.3)

#--------------------------------preprocessing--------------------------------------
def descaling(x_train,y_train,oversampling):
    pre=StandardScaler()
    x_train=pre.fit_transform(x_train)
    if(oversampling):
        ros=RandomOverSampler()
        x_train,y_train=ros.fit_resample(x_train,y_train)
    y_train=y_train.reshape(-1,1)
    data=np.hstack((x_train,y_train))
    return data,x_train,y_train
train,x_train,y_train=descaling(x_train,y_train,oversampling=True)
test,x_test,y_test=descaling(x_test,y_test,oversampling=False)
print(f"train:{train.shape},\nx_train:{x_train.shape},\ny_train{y_train.shape}")
print(f"test:{test.shape},\nx_test:{x_test.shape},\ny_test{y_test.shape}")
#-------------------------------model--------------------------------------

model=LogisticRegression()
model.fit(x_train,y_train)
model.fit(x_test,y_test)
pred=model.predict(x_test)
acc=accuracy_score(y_test,pred)*100

print(f"accuracy: {accuracy_score(y_test,pred)*100} %")
print(f"confusion matrix: {confusion_matrix(y_test,pred)}")
con=confusion_matrix(y_test,pred)
print(classification_report(y_test,pred))
acc=np.mean(acc)
ac1=np.mean(x_test)

def visulization():
    plt.figure(figsize=(10,6))
    plt.bar(['true positive','false postive','true negative','false positive'],[con[0][0],con[0][1],con[1][0],con[1][1]],color=['green','red','yellow','blue'])
    plt.figure(figsize=(10,6))
    plt.scatter((range(len(y_test))),y_test,color='blue',label='Actual')
    plt.scatter(range(len(pred)),pred,color='red',label='Predicted')
    plt.show()

    sns.heatmap(con, annot=True)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

visulization()




    
