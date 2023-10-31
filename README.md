
# CREDIT  CARD FRAUD DETECTION 

Credit card fraud detection is a set of methods and techniques designed to block fraudulent purchases, both online and in-store. This is done by ensuring that you are dealing with the right cardholder and that the purchase is legitimate.



## Acknowledgements

 - [Learn About ML using Python](https://www.geeksforgeeks.org/machine-learning-with-python/)
 - [Explore the Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfrauds)
 - [Algorithm using in Project](https://www.geeksforgeeks.org/k-nearest-neighbors-with-python-ml/)

## Features
Selecting and engineering pertinent features from the dataset is necessary to create a credit card fraud detection model that works. Here's a step-by-step tutorial on selecting and preparing features for Python credit card fraud detection:


- **Import Libraries and Load Data:**

Importing the required libraries and loading your dataset of credit card transactions should come first. For this purpose, you can utilise libraries like scikit-learn, numpy, and pandas.

    import pandas as pd 
    import numpy as np  
    #Load your dataset 
    data = pd.read_csv("creditcard.csv")

- **Data Exploration:** 

Investigate your dataset to learn about its composition, feature distribution, and balance between legitimate and fraudulent transactions. For this, visualisation programmes like Seaborn and Matplotlib can be useful.



    import matplotlib.pyplot as plt 
    import seaborn as sns 
    #Explore the data

    print(data.head()
    print(data.info())
    print(data["Class"].value_counts())
    # Visualize data distributions
    sns.countplot(data["Class"])
     plt.show()

  - **Feature Selection**
  Determine which features are most important for detecting fraud. Univariate feature selection, feature importance from tree-based models, and domain expertise are a few popular methods.

- Feature selection using feature importance from a tree-based model .

     (e.g., Random Forest)

        from sklearn.ensemble import RandomForestClassifier

         X = data.drop("Class", axis=1)
         y = data["Class"]
        model = RandomForestClassifier()
      model.fit(X, y)
      feature_importance = model.feature_importances_ 
      features = X.columns

       # Select the top N important features
       N = 10
       selected_features = features[np.argsort(feature_importance)[::-1][:N]] 
       print("Selected Features:", selected_features)

- **Feature Engineering:**
Develop new features or preprocess current ones to improve the model's fraud detection capabilities. This may consist of
use StandardScaler or MinMaxScaler, for example, to scale numerical features to equivalent scales.
if any, encoding category features.
constructing aggregations or interaction characteristics to identify trends in the data.

        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import LabelEncoder
        #Scale numerical features
        scaler = StandardScaler()
        scaler = StandardScaler()scaler.fit_transform(X[selected_features])
        #Encode categorical features (if any)
        #For example, if you have a "category" column:
        #encoder = LabelEncoder()
        #X["category_encoded"] = encoder.fit_transform(X["category"])

  
- **Data Splitting:**

 Split the data into training and testing sets to train and evaluate your credit card fraud detection model. 

  

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


- **Modeling:**

  Using the features that have been carefully chosen and constructed, train a aud.machine learning or deep learning model that is suitable for detecting credit card fraud.



#Example using a Random Forest classifier

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    #Make predictions

    y_pred = model.predict(X_test)

**Model Evaluation:**

Assess the model's performance using relevant metrics, such as accuracy, precision, recall, F1-score, and area under the ROC curve (AUC-ROC).



    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score 
    ("Classification Report:")

    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")

    print(confusion_matrix(y_test, y_pred))

    print("AUC-ROC Score:", roc_auc_score(y_test, y_pred))

Remember that feature engineering is an iterative process, and you may need to fine-tune your feature selection and engineering techniques based on your specific dataset and the performance of your model.

  


## Authors

- [@Md Nurullah](https://github.com/SheikhNoor)


  




    


     



  

  
     


 
