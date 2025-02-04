#Import Libraries 
import pandas as pd 
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt 

#for Encoding and Scaling of Data 
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder, MinMaxScaler

# For Prediction 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report



#Import Data 

mydata = pd.read_csv("thyroid_cancer_risk_data.csv")
df = pd.DataFrame(mydata)

# EDA
print("===================================# VIEWING AND INSPECTING DATA =================================== \n")
print("---------- # FIRST 5 ROWS # -----------\n",df.head())
print("---------- #  LAST 5 ROWS# # -----------\n",df.tail())
print("---------- #  SUMMARY OF THE DATAFRAME# # -----------\n",df.info())
print("---------- #  STATISTICS OF NUMERICAL COLUMNS# # -----------\n",df.describe().T)
print("---------- #  SHAPE OF THE DATAFRAME# # -----------\n",df.shape)
print("---------- #  LIST OF COLUMN NAMES# # -----------\n",df.columns)
print("---------- #  DATA TYPES OF COLUMNS# # -----------\n",df.dtypes)

# Checking Null Columns
print("---------- #  NULL VALUES INFO # -----------\n",df.isnull().sum())
 
#Columns where the Null Values are existing 

null_counts = df.isnull().sum()
null_columns = null_counts[null_counts > 0].index.tolist()
#Object Columns with Null Values
object_columns = [col for col in null_columns if df[col].dtype == 'object']
#Int / float Columns with Null Values
int_columns = [col for col in null_columns if df[col].dtype in ['int64','float64']]

print (f"Object Columns with Null Values:  {object_columns}")
print (f"Non Object Columns with Null Values:  {int_columns}")

# Fill Object Columns 
for col in object_columns:
    df[col] =df[col].fillna("No Data")
    print(f"NULL VALUE IS UPDATED FOR {col}")


# Encoders
Label_Encoder = LabelEncoder() 
Ordinal_Encoder = OrdinalEncoder()
# Classifying Nominal and Ordinal Columns
Nominal_Columns =['Gender', 'Country','Ethnicity','Family_History','Radiation_Exposure','Iodine_Deficiency','Smoking','Obesity','Diabetes','Diagnosis']
Ordinal_Columns =['Thyroid_Cancer_Risk']

# Encoding Data
print(" \n NOW DATA WILL BE ENCODED \n")
for col in Nominal_Columns:
    df[col] =Label_Encoder.fit_transform(df[col])
    print(f"ENCODED >>> {col}")

for col in Ordinal_Columns:
    df[col]= Ordinal_Encoder.fit_transform(df[[col]])
    print(f"ENCODED >>> {col}")

# Saving Cleaned Data to a new file 
df.to_csv('thyroid_cancer_risk_data_cleaned.csv',index=False)
print(">>> NEW FILE CREATED WITH CLEAN DATA")

#Visualization 


plt.figure(figsize=(15,10))
plt.subplot(2,3,1)
sns.histplot(df.iloc[:,0], kde=True)
plt.title("Histogram")

plt.subplot(2,3,2)
sns.scatterplot(data=df.head(200), x='Age', y='Ethnicity', hue='Gender')
plt.title("Scatterplot with 200 Sample")

plt.subplot(2,3,3)
sns.lineplot(data=df.head(200), x='Country', y='Family_History', hue='Gender')
plt.title("Lineplot for 200 sample")

plt.subplot(2,3,4)
sns.boxplot(data=df, x='Obesity', y='Gender')
plt.title("Boxplot")
    
plt.subplot(2,3,5)
sns.heatmap(df.head(50).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap with sample of 50")
    
plt.subplot(2,3,6)
sns.countplot(x=df.head(50).columns[0], data=df.head(50))
plt.title("Countplot with sample of 50")
    
plt.show()


# features =['Age','Gender','Family_History','Radiation_Exposure','Iodine_Deficiency','Smoking','Obesity','Diabetes',
#            'TSH_Level','T3_Level','T4_Level','Nodule_Size']

X = df.iloc[:,:-1]
y= df['Diagnosis']

#split the data 
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#scale the data
scaler = MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#selecting the model

# model=RandomForestClassifier() # Accuracy Score is ~82%
model=AdaBoostClassifier() # Accuracy Score is ~82%
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
