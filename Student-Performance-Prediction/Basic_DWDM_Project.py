import pandas as pd 
    
# making dataframe 
data = pd.read_excel(r'D:\Academics\5th SEM\DMPA LAB\Filtered_student.xlsx') 
# output the dataframe
#print(df)


print(data.head())


# Import train_test_split function
from sklearn.model_selection import train_test_split

X=data[['school','sex', 'address','famsize','Pstatus','Mjob','Fjob','guardian','schoolsup','famsup', 'paid','activities','nursery','higher','internet','romantic','age','Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout','health','Dalc','Walc','G1','G2','G3' ]]  # Features
y=data['health']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)



RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)


import pandas as pd
feature_name = list(data.columns)
#print(feature_name)
feature_name.remove('health')
#print(feature_name)
feature_imp = pd.Series(clf.feature_importances_,index=feature_name).sort_values(ascending=False)
feature_imp


import seaborn
seaborn.heatmap(data.corr(),
                xticklabels=data.columns,
                yticklabels=data.columns)


import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

z = [1,0,0,0,0,0,1,0,1,0,0,0,1,1,0,0,18,4,4,2,2,0,4,3,4,5,1,6,5,6,6]
k = clf.predict([z])
print(k)
