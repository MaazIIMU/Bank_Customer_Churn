import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# Fetch the data
data = pd.read_csv('/Users/maaz/Documents/Academic Resources/Learning Tools/Python Data Analysis/Bank customer churn/Customer-Churn-Records.csv')

# Exploring the data, finding null values if applicable
def explore_data(data):
    print(data.head())
    print(data.info())
    print(data.describe())
    print(data.isnull().sum())


# Find out the average age of customers
def average_age(data):
    avg_age = round(data['Age'].mean(),0)
    print('-'*30)
    print("Average Age = ",avg_age)
    print('-'*30)

# Out of the dataset, how many customers have churned?
def churned_customers(data):
    churned_customers = data['Exited'].value_counts()[1]
    total_customers = data.shape[0]
    churn_rate = round((churned_customers/total_customers)*100,2)
    print('-'*30)
    print("Number of customers who have churned = ",churned_customers)
    print("Total number of customers = ",total_customers)
    print("Churn Rate = ",churn_rate,"%")
    print('-'*30)

# What are the attributes of customers who have churned?
def attributes_of_churned_customers(data):
    churned_data = data[data['Exited'] == 1]
    churned_avg_age = round(churned_data['Age'].mean(),0)
    churned_avg_credit_score = round(churned_data['CreditScore'].mean(),0)
    churned_avg_balance = round(churned_data['Balance'].mean(),2)
    churned_avg_tenure = round(churned_data['Tenure'].mean(),0)
    churned_had_credit_card = round(churned_data['HasCrCard'].mean()*100,2)
    churned_avg_s_score = round(churned_data['Satisfaction Score'].mean(),2)
    churned_avg_points = round(churned_data['Point Earned'].mean(),2)
    churned_men_count = (churned_data['Gender'] == 'Male').sum()
    churned_customers = data['Exited'].value_counts()[1]
    churned_males = round((churned_men_count/churned_customers*100),0)
    print('-'*30)
    print("Attributes of customers who have churned:")
    print("Average Age = ",churned_avg_age)
    print("Average Credit Score = ",churned_avg_credit_score)
    print("Average Balance = ",churned_avg_balance)
    print("Average Tenure = ",churned_avg_tenure)
    print("Percentage of customers who had a credit card = ",churned_had_credit_card,"%")
    print("Average Satisfaction Score = ",churned_avg_s_score)
    print("Average Points Earned = ",churned_avg_points)
    print("Proportion of men = ", churned_males, "%")
    print('-'*30)

# Attributes of customers who have not churned
def attributes_of_non_churned_customers(data):
    non_churned_data = data[data['Exited'] == 0]
    non_churned_avg_age = round(non_churned_data['Age'].mean(),0)
    non_churned_avg_credit_score = round(non_churned_data['CreditScore'].mean(),0)
    non_churned_avg_balance = round(non_churned_data['Balance'].mean(),2)
    non_churned_avg_tenure = round(non_churned_data['Tenure'].mean(),0)
    non_churned_had_credit_card = round(non_churned_data['HasCrCard'].mean()*100,2)
    non_churned_avg_s_score = round(non_churned_data['Satisfaction Score'].mean(),2)
    non_churned_avg_points = round(non_churned_data['Point Earned'].mean(),2)
    non_churned_m = round((non_churned_data['Gender'] == 'Male').sum()/non_churned_data.shape[0]*100,0)
    print('-'*30)
    print("Attributes of customers who have not churned:")
    print("Average Age = ",non_churned_avg_age)
    print("Average Credit Score = ",non_churned_avg_credit_score)
    print("Average Balance = ",non_churned_avg_balance)
    print("Average Tenure = ",non_churned_avg_tenure)
    print("Percentage of customers who had a credit card = ",non_churned_had_credit_card,"%")
    print("Average Satisfaction Score = ",non_churned_avg_s_score)
    print("Average Points Earned = ",non_churned_avg_points)
    print("Proportion of men = ", non_churned_m, "%")
    print('-'*30)

# Plot the distribution of customers by geography and churn status

def plot_geography (data):
    sns.countplot(
        x='Geography', 
        hue='Exited', 
        data=data,
        palette='Blues'
    )
    plt.title('Distribution of Customers by Geography and Churn Status', fontweight='bold')
    plt.xlabel('Geography', fontweight='bold')
    plt.ylabel('Number of Customers', fontweight='bold')
    plt.savefig('geography_churn_distribution.png', dpi=300)
    plt.show()

# Run a logistic regression to find out the significant predictors of customer churn
def logistic_regression(data):
    data = pd.get_dummies(data, drop_first=True)
    X = data[['CreditScore', 'Geography_Germany', 'Geography_Spain', 'Gender_Male', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Satisfaction Score', 'Point Earned']]
    X = X.astype(float)
    y = data['Exited']
    X = sm.add_constant(X)
    model = sm.Logit(y, X)
    result = model.fit()
    print(result.summary())

# Create a predictive model using logistic regression and evaluate its performance
def predictive_model(data):
    data = data.drop(['CustomerId', 'Surname','RowNumber'], axis=1)
    data = pd.get_dummies(data, drop_first=True)
    X = data.drop('Exited', axis=1)
    y = data['Exited']
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print('-'*30)
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print('-'*30)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print('-'*30)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print('-'*30)
    print("ROC AUC Score:", round(auc,4))
    print('-'*30)

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.plot(fpr,tpr)
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')   
    plt.title('ROC Curve', fontweight='bold')
    plt.savefig('roc_curve.png', dpi=300)
    plt.show()

# Run the functions
explore_data(data)
# average_age(data)
# churned_customers(data)
# attributes_of_churned_customers(data)
# attributes_of_non_churned_customers(data)
# plot_geography(data)
# logistic_regression(data)
# predictive_model(data)

