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