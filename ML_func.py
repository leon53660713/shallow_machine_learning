# import package
# warning
import warnings
warnings.filterwarnings("ignore")
# data
import numpy as np
import pandas as pd
# visulaize
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
# adjust data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# ML
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
# PCA
from sklearn.decomposition import PCA
# evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, classification_report, ConfusionMatrixDisplay


class ml_func:
    # init
    def __init__(self):
        warnings.filterwarnings("ignore")
    # def preprocess_data
    def preprocess_data(self, X, y, seed, standard=False, pca=False, n_pc=None, show_plot=False):
        # spilt data
#         X = df.drop(columns=[y_col])
#         y = df[y_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed)

        # standardlize
        if standard:
            # standardlize train/test
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # show num of standardlize train/test
            print("standardlize training data :", len(X_train_scaled))
            print("standardlize testing data :", len(X_test_scaled))

            X_train = X_train_scaled
            X_test = X_test_scaled

        # pca
        if pca:
            # pca
            pca_model = PCA(n_components=n_pc)
            X_train = pca_model.fit_transform(X_train)
            X_test = pca_model.transform(X_test)

            # prompt message
            print("finish doing pca")

            # if want to show plot
            if show_plot:
                # calculate
                eigvals = pca_model.explained_variance_
                sum_eigvals_ratio = np.cumsum(pca_model.explained_variance_ratio_)

                # plot
                fig, ax1 = plt.subplots(figsize=(8, 5))
                # ax1--bar
                x1 = range(1, len(eigvals) + 1)
                color1 = "steelblue"
                ax1.bar(x1, eigvals, align='center', color=color1)
                ax1.set_xlabel("principal component", fontsize=10)
                ax1.set_ylabel("variance explained", color=color1, fontsize=10)
                ax1.tick_params(axis='y', labelcolor=color1)
                ax1.tick_params(axis="both", labelsize=10)
                # set x labels
                ax1.set_xticks(x1)
                ax1.set_xticklabels(["PC{}".format(i) for i in range(1, len(eigvals) + 1)], fontsize=8)
                ax1.set_title('pareto plot', fontsize=15)
                # ax2--line
                ax2 = ax1.twinx()
                color2 = "brown"
                ax2.plot(x1, sum_eigvals_ratio*100, color=color2, marker='s', linestyle='-', linewidth=2, markersize=5)
                ax2.set_ylabel("cumulative variance explained (%)", color=color2, fontsize=10)
                ax2.tick_params(axis="y", labelcolor=color2, labelsize=10)
                ax2.yaxis.set_major_formatter(PercentFormatter())
                ax2.tick_params(axis="both", labelsize=10)

                # show
                plt.grid(True)
                plt.show()

        if not standard and not pca:
            # show num of train/test
            print("training data :", len(X_train))
            print("testing data :", len(X_test))

        return X_train, X_test, y_train, y_test

    # def evaluate_model
    def evaluate_model(self, y_test, y_pred):
        # accuracy
        accuracy = round(accuracy_score(y_test, y_pred), 2)
        # precision
        precision = round(precision_score(y_test, y_pred, average='weighted'), 2)
        # recall
        recall = round(recall_score(y_test, y_pred, average='weighted'), 2)
        # F1
        f1 = round(f1_score(y_test, y_pred, average='weighted'), 2)
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        # report
        report = classification_report(y_test, y_pred)

        # save to dict
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "F1-score": f1,
        }

        return results
