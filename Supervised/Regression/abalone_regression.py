import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


dataset_info = """
Abalone Dataset
Predicting the age of abalone from physical measurements.  
The age of abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope -- a boring and time-consuming task.
Other measurements, which are easier to obtain, are used to predict the age.
Further information, such as weather patterns and location (hence food availability) may be required to solve the problem.
From the original data examples with missing values were removed (the majority having the predicted value missing), and the ranges of the continuous values have been scaled for use with an ANN (by dividing by 200).
"""


def corr_plot(df):
    ndf = df.drop("Sex", axis=1)
    corr = ndf.corr()
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    sns.set_theme(style="white")
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix with Values")
    plt.savefig("./images/correlation_matrix.png", dpi=300, bbox_inches='tight')
    # plt.show()


def pair_plot(df):
    ndf = df.drop("Sex", axis=1)
    plt.figure(figsize=(8, 6)) # Adjust figure size as needed
    sns.set_theme(style="ticks")
    sns.pairplot(ndf,  hue="Rings", palette="pastel")
    plt.savefig("./images/pair_plot.png", dpi=300, bbox_inches='tight')
    # plt.show()


def box_plot(df):
    ndf = df.drop("Sex", axis=1)    
    nndf = ndf.drop('Rings', axis=1)
    cols = nndf.columns.tolist()

    fig, ax = plt.subplots(figsize=(16, 9))

    ax.boxplot(nndf, 
                patch_artist=True, showmeans=True, notch=True,
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='red', linewidth=2),
                meanprops=dict(marker='o', markeredgecolor='black', markerfacecolor='green'),
            # label=cols
            )

    ax.set_xticklabels(cols)
    plt.title("Box Plot of Features")
    plt.savefig("./images/box_plot.png", dpi=300, bbox_inches='tight')
    # plt.show()


def X_y_split(df):
    X = df.drop(columns=["Sex", "Rings"], axis=1).values
    y = df["Rings"].values
    y = y.reshape(1,-1).transpose()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
    

def linear_regression(X_train, X_test, y_train, y_test):
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Linear Regression MSE: {mse:.2f}, R^2: {r2:.2f}")
    
    return model, mse, r2


def ridge_regression(X_train, X_test, y_train, y_test):
    model = linear_model.Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Ridge Regression MSE: {mse:.2f}, R^2: {r2:.2f}")
    
    return model, mse, r2


def lasso_regression(X_train, X_test, y_train, y_test):
    model = linear_model.Lasso(alpha=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Lasso Regression MSE: {mse:.2f}, R^2: {r2:.2f}")
    
    return model, mse, r2


def elastic_net_regression(X_train, X_test, y_train, y_test):
    model = linear_model.ElasticNet(alpha=0.1, l1_ratio=0.5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Elastic Net Regression MSE: {mse:.2f}, R^2: {r2:.2f}")
    
    return model, mse, r2   


def decision_tree_regression(X_train, X_test, y_train, y_test):
    model = tree.DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Decision Tree Regression MSE: {mse:.2f}, R^2: {r2:.2f}")
    
    return model, mse, r2


def random_forest_regression(X_train, X_test, y_train, y_test): 
    model = ensemble.RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Random Forest Regression MSE: {mse:.2f}, R^2: {r2:.2f}")
    
    return model, mse, r2


def svr_regression(X_train, X_test, y_train, y_test):
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"SVR Regression MSE: {mse:.2f}, R^2: {r2:.2f}")
    
    return model, mse, r2


def rfe_regression(X_train, X_test, y_train, y_test):
    model = linear_model.LinearRegression()
    rfe = RFE(model, n_features_to_select=5)
    rfe.fit(X_train, y_train.ravel())
    
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)

    model.fit(X_train_rfe, y_train)
    y_pred = model.predict(X_test_rfe)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"RFE Regression MSE: {mse:.2f}, R^2: {r2:.2f}")
    
    return model, mse, r2


def gboost_regression(X_train, X_test, y_train, y_test):
    params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
    }

    model = ensemble.GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_pred, y_test)
    r2 = r2_score(y_test, y_pred)

    print(f"GBoost Regression MSE: {mse:.2f}, R^2: {r2:.2f}")

    return model, mse, r2


def torch_regression(X_train, X_test, y_train, y_test, num_epochs = 1000):

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=y_test_tensor.shape[0], shuffle=False)


    class ANNRegration(nn.Module):
        def __init__(self, input_size):
            super(ANNRegration,self).__init__()
            self.h1 = nn.Linear(input_size, 32)
            self.relu = nn.ReLU()
            self.out = nn.Linear(32,1)

        def forward(self, x):
            x = self.h1(x)
            x = self.relu(x)
            x = self.out(x)
            return x
        
    model = ANNRegration(7)
    MSELoss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1000

    for epoch in range(num_epochs):
        for i,(X, y) in enumerate(train_loader):
            optimizer.zero_grad()

            y_hat = model(X)

            loss = MSELoss(y, y_hat)

            loss.backward()
            optimizer.step()

    
    model.eval()
    mse = 0
    r2 = 0

    with torch.no_grad():
        # ann_mse = 0
        # ann_r2 = 0
        for X, y in test_loader:
            y_hat = model(X)
            mse += MSELoss(y, y_hat).item()
            r2 += r2_score(y, y_hat)
            
        mse/=len(test_loader)
        r2 /= len(test_loader)

    print(f"ANN Regression MSE: {mse:.2f}, R^2: {r2:.2f}")
    
    return model, mse, r2



if __name__ == "__main__":
    df = pd.read_csv("./data/abalone/abalone.csv")

    print(dataset_info, "\n")
    print(df.head(), "\n")
    print(df.describe().T)

    if not os.path.exists("./images/correlation_matrix.png"):
        corr_plot(df)

    if not os.path.exists("./images/pair_plot.png"):
        pair_plot(df)

    if not os.path.exists("./images/box_plot.png"):
        box_plot(df)

    X_train, X_test, y_train, y_test = X_y_split(df)
    lr_model, lr_mse, lr_r2 = linear_regression(X_train, X_test, y_train, y_test)
    rg_model, rg_mse, rg_r2 = ridge_regression(X_train, X_test, y_train, y_test)
    la_model, la_mse, la_r2 = lasso_regression(X_train, X_test, y_train, y_test)
    elastic_model, elastic_mse, elastic_r2 = elastic_net_regression(X_train, X_test, y_train, y_test)
    dt_model, dt_mse, dt_r2 = decision_tree_regression(X_train, X_test, y_train, y_test)
    rf_model, rf_mse, rf_r2 = random_forest_regression(X_train, X_test, y_train, y_test)
    svr_model, svr_mse, svr_r2 = svr_regression(X_train, X_test, y_train, y_test)
    rfe_model, rfe_mse, rfe_r2 = rfe_regression(X_train, X_test, y_train, y_test)
    gboost_model, gboost_mse, gboost_r2 = gboost_regression(X_train, X_test, y_train, y_test)
    torch_model, torch_mse, torch_r2 = torch_regression(X_train, X_test, y_train, y_test)


    results={
    "Linear": [lr_mse, lr_r2],
    "Ridge ": [rg_mse, rg_r2],
    "Lasso ": [la_mse, la_r2],
    "Elastic Net ": [elastic_mse, elastic_r2],
    "Decision Tree": [dt_mse, dt_r2],
    "Random Forest ": [rf_mse, rf_r2],
    "SVR": [svr_mse, svr_r2],
    "GBoost": [gboost_mse, gboost_r2],
    "ANN" :[torch_mse, torch_r2]
    }

    print(pd.DataFrame(results, index=["MSE", "R2_Score"]))


