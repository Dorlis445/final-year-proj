import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import joblib

df = pd.read_csv('sample.csv')

columns = ['DailyRate', 'EducationField', 'EmployeeCount', 'EmployeeNumber', 'HourlyRate', 'MonthlyRate',
        'Over18', 'RelationshipSatisfaction', 'StandardHours']
df.drop(columns, inplace=True, axis=1)

labeled_columns = ['Attrition', 'BusinessTravel', 'Department',
                      'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
from sklearn import preprocessing
data_encoded = df.copy(deep=True)
#Use Scikit-learn label encoding to encode character data
lab_enc = preprocessing.LabelEncoder()
for col in labeled_columns:
        data_encoded[col] = lab_enc.fit_transform(df[col])
        le_name_mapping = dict(zip(lab_enc.classes_, lab_enc.transform(lab_enc.classes_)))
        print('Feature', col)
        print('mapping', le_name_mapping)

independent_data = data_encoded.drop(['Attrition'], axis=1)
dependent_data = data_encoded[['Attrition']]

selected_data = data_encoded[['MonthlyIncome','EnvironmentSatisfaction','PercentSalaryHike','Age', 'YearsSinceLastPromotion','JobRole','PerformanceRating', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole','JobSatisfaction','YearsWithCurrManager','WorkLifeBalance', 'JobInvolvement',
                     'JobLevel',  'Attrition']]

independent_data = selected_data.drop(['Attrition'], axis=1)
dependent_data = selected_data[['Attrition']] 

X = independent_data
y = dependent_data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)

standardized_data = scaler.transform(X)

X = standardized_data
y = dependent_data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
import numpy as np
input_data = (5130,3,23,49,1,6,4,10,10,7,2,7,3,2,2	)
#changing the input data into numpy array
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
std_data = scaler.transform(input_data_reshaped)


prediction = model.predict(std_data) 
print(prediction)

if (prediction[0] == 1):
    print('Employee likely to leave')
else:
    print('Employee not leaving')


import joblib
joblib.dump(model,'model.pkl')


model = joblib.load("model.pkl")




 