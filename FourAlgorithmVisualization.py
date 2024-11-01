import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pickle

st.subheader("Four algorithm visualization")
st.subheader("1. Logistic Regression")
st.subheader("2. K-nearest neighbour")
st.subheader("3. Support Vector Machine")
st.subheader("4. Density-Based Spatial Clustering of Applications with Noise")

Algorithm = st.selectbox(
    "Select Algorithm",
    ("Select Algorithm","LogisticRegression", "K-NearestNeighbour","SupportVectorMachine","Density-Based Spatial Clustering of Applications with Noise")
)

if Algorithm == "LogisticRegression":
    # Dataset graph(according the type of graph)
    def load_initial_graph(dataset, ax, centers):
        if dataset == "Binary":
            x, y = make_blobs(n_features=2, centers=centers, random_state=6)
            ax.scatter(x.T[0], x.T[1], c=y, cmap='rainbow')
            return x, y
        elif dataset == "Multiclass":
            x, y = make_blobs(n_features=2, centers=centers, random_state=6)
            ax.scatter(x.T[0], x.T[1], c=y, cmap='rainbow')
            return x, y


    # draw_meshgrid function
    def draw_meshgrid():
        a = np.arange(start=x[:, 0].min() - 1, stop=x[:, 0].max() + 1, step=0.01)
        b = np.arange(start=x[:, 1].min() - 1, stop=x[:, 1].max() + 1, step=0.01)

        XX, YY = np.meshgrid(a, b)
        input_array = np.array([XX.ravel(), YY.ravel()]).T
        return XX, YY, input_array


    # Sidebar heading
    st.sidebar.markdown("# Logistic Regression")

    # Sidebar options(selectbox, number_input)
    # Dataset type
    Dataset = st.sidebar.selectbox(
        'Select Dataset',
        ('Binary', 'Multiclass')
    )

    # Penalty(l1, l2, elasticnet)
    Penalty = st.sidebar.selectbox(
        "Select Penalty",
        ('l1', 'l2', 'elasticnet', 'none')
    )

    # Value of c
    C = st.sidebar.number_input("Enter C", value=1.0)

    # Solver(lbfgs, liblinear, newton-cg, newton-cholesky, sag, saga)
    Solver = st.sidebar.selectbox(
        "Select Solver",
        ('lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga')
    )

    # Number of iterations
    Iteration = st.sidebar.number_input("Enter number of iteration", step=1, format="%d", min_value=0)

    # Multi class(auto, ovr, multinomial)
    Multi_Class = st.sidebar.selectbox(
        "Enter Multi_Class",
        ('auto', 'ovr', 'multinomial')
    )

    # Value of l1_ratio
    L1_Ratio = st.sidebar.number_input("Enter L1 Ratio", min_value=0)

    # Graph plot
    fig, ax = plt.subplots()

    # Condition of binary or multiclass dataset
    if Dataset == "Multiclass":
        centers = st.number_input("Enter centers", min_value=2, format="%d")
        x, y = load_initial_graph(Dataset, ax, centers)
    else:
        x, y = load_initial_graph(Dataset, ax, 2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    orig = st.pyplot(fig)

    # Button to run algorithm
    if st.sidebar.button('Run Algorithm'):
        # first clear graph(on the time of run)
        orig.empty()

        # Algorithm(parameters, train, test)
        clf = LogisticRegression(penalty=Penalty, C=C, solver=Solver, max_iter=Iteration, multi_class=Multi_Class,
                                 l1_ratio=L1_Ratio)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        XX, YY, input_array = draw_meshgrid()
        labels = clf.predict(input_array)

        # Parameters(with values
        st.subheader("Parameters(with values)")
        st.write("Penalty: ", Penalty)
        st.write("C: ", C)
        st.write("solver: ", Solver)
        st.write("max_iter: ", Iteration)
        st.write("multi_class: ", Multi_Class)
        st.write("l1_ratio: ", L1_Ratio)

        # Result as a graph(also print accuracy of logistic regression)
        ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        origin = st.pyplot(fig)
        st.subheader("Accuracy for Logistic Regression:" + str(round(accuracy_score(y_test, y_pred), 2)))
elif Algorithm == "K-NearestNeighbour":
    st.sidebar.markdown("# K-nearest neighbors")
    Dataset = st.sidebar.selectbox(
        'Select Dataset',
        ('Select Dataset', 'Iris', 'Wine Quality', 'Placement', 'Student Performance')
    )

    # Iris Dataset
    if Dataset == 'Iris':
        iris = pickle.load(open("iris.pkl", "rb"))

        sepallength = st.sidebar.number_input("Enter sepal length", min_value=0.0)
        sepalwidth = st.sidebar.number_input("Enter sepal width", min_value=0.0)
        petallength = st.sidebar.number_input("Enter petal length", min_value=0.0)
        petalwidth = st.sidebar.number_input("Enter petal width", min_value=0.0)
        k = st.sidebar.number_input("Enter k", min_value=1, format="%d")

        distance = []
        for i in range(iris.shape[0]):
            value1 = ((iris['sepal.length'][i] - sepallength) ** 2) + ((iris['sepal.width'][i] - sepalwidth) ** 2)
            value2 = ((iris['petal.length'][i] - petallength) ** 2) + ((iris['petal.width'][i] - petalwidth) ** 2)
            value = np.sqrt(value1 + value2)
            distance.append(value)

        m = 999
        iteration = 1
        lis1 = []
        lis2 = []
        lis3 = []
        lis4 = []
        lis5 = []
        while iteration <= k:
            sepall = 0
            sepalw = 0
            petall = 0
            petalw = 0
            variety = 0
            index = 0
            for i in range(iris.shape[0]):
                if m > distance[i]:
                    m = distance[i]
                    sepall = iris['sepal.length'][i]
                    sepalw = iris['sepal.width'][i]
                    petall = iris['petal.length'][i]
                    petalw = iris['petal.width'][i]
                    variety = iris['variety'][i]
                    index = i
            lis1.append(sepall)
            lis2.append(sepalw)
            lis3.append(petall)
            lis4.append(petalw)
            lis5.append(variety)
            distance[index] = 999
            m = 999
            iteration = iteration + 1

        if st.sidebar.button("Run Algorithm"):
            st.subheader("Table of k nearest points")
            st.write("Value of k:", k)
            data = {"sepallength": lis1, "sepalwidth": lis2, "petallength": lis3, "petalwidth": lis4, "variety": lis5}
            data = pd.DataFrame(data)
            data.index = data.index + 1

            column = np.array(data['variety'])
            value = {"Setosa": 0, "Versicolor": 0, "Virginica": 0}
            for i in range(data.shape[0]):
                if column[i] == "Setosa":
                    value['Setosa'] = value['Setosa'] + 1
                elif column[i] == "Versicolor":
                    value['Versicolor'] = value['Versicolor'] + 1
                elif column[i] == "Virginica":
                    value['Virginica'] = value['Virginica'] + 1

            m = -999
            answer = ""
            for i in value:
                if m < value[i]:
                    m = value[i]
                    answer = i

            st.table(data)
            st.write("Output variety is: ", answer)

    # Wine Quality Dataset
    if Dataset == "Wine Quality":
        wineqt = pickle.load(open("wineqt.pkl", "rb"))

        fixedacidity = st.sidebar.number_input("Enter fixed acidity", min_value=0.0)
        volatileacidity = st.sidebar.number_input("Enter volatile acidity", min_value=0.0)
        citricacid = st.sidebar.number_input("Enter citric acid", min_value=0.0)
        residualsugar = st.sidebar.number_input("Enter residual sugar", min_value=0.0)
        chlorides = st.sidebar.number_input("Enter chlorides", min_value=0.0)
        freesulfurdioxide = st.sidebar.number_input("Enter free sulfur dioxide", min_value=0.0)
        totalsulfurdioxide = st.sidebar.number_input("Enter total sulfur dioxide", min_value=0.0)
        density = st.sidebar.number_input("Enter density", min_value=0.0)
        pH = st.sidebar.number_input("Enter pH", min_value=0.0)
        sulphates = st.sidebar.number_input("Enter sulphates", min_value=0.0)
        alcohol = st.sidebar.number_input("Enter alcohol", min_value=0.0)
        k = st.sidebar.number_input("Enter k", min_value=1)

        distance = []
        for i in range(wineqt.shape[0]):
            value1 = ((wineqt['fixed acidity'][i] - fixedacidity) ** 2) + (
                        (wineqt['volatile acidity'][i] - volatileacidity) ** 2) + (
                                 (wineqt['citric acid'][i] - citricacid) ** 2) + (
                                 (wineqt['residual sugar'][i] - residualsugar) ** 2)
            value2 = ((wineqt['chlorides'][i] - chlorides) ** 2) + (
                        (wineqt['free sulfur dioxide'][i] - freesulfurdioxide) ** 2) + (
                                 (wineqt['total sulfur dioxide'][i] - totalsulfurdioxide) ** 2) + (
                                 (wineqt['density'][i] - density) ** 2)
            value3 = ((wineqt['pH'][i] - pH) ** 2) + ((wineqt['sulphates'][i] - sulphates) ** 2) + (
                        (wineqt['alcohol'][i] - alcohol) ** 2)
            value = np.sqrt(value1 + value2 + value3)
            distance.append(value)

        m = 999
        iteration = 1
        lis1 = []
        lis2 = []
        lis3 = []
        lis4 = []
        lis5 = []
        lis6 = []
        lis7 = []
        lis8 = []
        lis9 = []
        lis10 = []
        lis11 = []
        lis12 = []
        while iteration <= k:
            fixedacidity = 0
            volatileacidity = 0
            citricacid = 0
            residualsugar = 0
            chlorides = 0
            freesulfurdioxide = 0
            totalsulfurdioxide = 0
            density = 0
            pH = 0
            sulphates = 0
            alcohol = 0
            quality = -1
            index = 0
            for i in range(wineqt.shape[0]):
                if m > distance[i]:
                    m = distance[i]
                    fixedacidity = wineqt["fixed acidity"][i]
                    volatileacidity = wineqt["volatile acidity"][i]
                    citricacid = wineqt["citric acid"][i]
                    residualsugar = wineqt["residual sugar"][i]
                    chlorides = wineqt["chlorides"][i]
                    freesulfurdioxide = wineqt["free sulfur dioxide"][i]
                    totalsulfurdioxide = wineqt["total sulfur dioxide"][i]
                    density = wineqt["density"][i]
                    pH = wineqt["pH"][i]
                    sulphates = wineqt["sulphates"][i]
                    alcohol = wineqt["alcohol"][i]
                    quality = wineqt["quality"][i]
                    index = i
            lis1.append(fixedacidity)
            lis2.append(volatileacidity)
            lis3.append(citricacid)
            lis4.append(residualsugar)
            lis5.append(chlorides)
            lis6.append(freesulfurdioxide)
            lis7.append(totalsulfurdioxide)
            lis8.append(density)
            lis9.append(pH)
            lis10.append(sulphates)
            lis11.append(alcohol)
            lis12.append(quality)
            distance[index] = 999
            m = 999
            iteration = iteration + 1

        if st.sidebar.button("Run Algorithm"):
            st.subheader("Table of k nearest points")
            st.write("Value of k:", k)
            data = {"fixed acidity": lis1, "volatile acidity": lis2, "citric acid": lis3, "residual sugar": lis4,
                    "chlorides": lis5, "free sulfur dioxide": lis6, "total sulfur dioxide": lis7, "density": lis8,
                    "pH": lis9, "sulphates": lis10, "alcohol": lis11, "quality": lis12}
            data = pd.DataFrame(data)
            data.index = data.index + 1

            column = np.array(data['quality'])
            value = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0, "10": 0}
            for i in range(data.shape[0]):
                if column[i] == 0:
                    value['0'] = value['0'] + 1
                elif column[i] == 1:
                    value['1'] = value['1'] + 1
                elif column[i] == 2:
                    value['2'] = value['2'] + 1
                elif column[i] == 3:
                    value['3'] = value['3'] + 1
                elif column[i] == 4:
                    value['4'] = value['4'] + 1
                elif column[i] == 5:
                    value['5'] = value['5'] + 1
                elif column[i] == 6:
                    value['6'] = value['6'] + 1
                elif column[i] == 7:
                    value['7'] = value['7'] + 1
                elif column[i] == 8:
                    value['8'] = value['8'] + 1
                elif column[i] == 9:
                    value['9'] = value['9'] + 1
                elif column[i] == 10:
                    value['10'] = value['10'] + 1

            m = -999
            answer = ""
            for i in value:
                if m < value[i] and value[i] != 0:
                    m = value[i]
                    answer = i

            st.table(data)
            st.write("Output quality is: ", answer)

    # Placement Dataset
    if Dataset == 'Placement':
        Placement = pickle.load(open("placement.pkl", "rb"))

        cgpa = st.sidebar.number_input("Enter cgpa", min_value=0.0)
        placement_exam_marks = st.sidebar.number_input("Enter placement exam marks", min_value=0.0)
        k = st.sidebar.number_input("Enter k", min_value=1, format="%d")

        distance = []
        for i in range(Placement.shape[0]):
            value = ((Placement['cgpa'][i] - cgpa) ** 2) + (
                        (Placement['placement_exam_marks'][i] - placement_exam_marks) ** 2)
            value = np.sqrt(value)
            distance.append(value)

        m = 999
        iteration = 1
        lis1 = []
        lis2 = []
        lis3 = []
        while iteration <= k:
            cgpa = 0
            placement_exam_marks = 0
            placed = -1
            index = 0
            for i in range(Placement.shape[0]):
                if m > distance[i]:
                    m = distance[i]
                    cgpa = Placement['cgpa'][i]
                    placement_exam_marks = Placement['placement_exam_marks'][i]
                    placed = Placement['placed'][i]
                    index = i
            lis1.append(cgpa)
            lis2.append(placement_exam_marks)
            lis3.append(placed)
            distance[index] = 999
            m = 999
            iteration = iteration + 1

        if st.sidebar.button("Run Algorithm"):
            st.subheader("Table of k nearest points")
            st.write("Value of k:", k)
            data = {"cgpa": lis1, "placement_exam_marks": lis2, "placed": lis3}
            data = pd.DataFrame(data)
            data.index = data.index + 1

            column = np.array(data['placed'])
            value = {"0": 0, "1": 0}
            for i in range(data.shape[0]):
                if column[i] == 0:
                    value['0'] = value['0'] + 1
                elif column[i] == 1:
                    value['1'] = value['1'] + 1

            m = -999
            answer = ""
            for i in value:
                if m < value[i]:
                    m = value[i]
                    answer = i

            st.table(data)
            st.write("Note: 0 means Not Placed")
            st.write("Note: 1 means  Placed")
            st.write("Output placed is: ", answer)

    # Student Performance Dataset
    if Dataset == 'Student Performance':
        StudentPerformance = pickle.load(open("StudentPerformance.pkl", "rb"))

        mathscore = st.sidebar.number_input("Enter math score", min_value=0)
        readingscore = st.sidebar.number_input("Enter reading score", min_value=0)
        writingscore = st.sidebar.number_input("Enter writing score", min_value=0)
        k = st.sidebar.number_input("Enter k", min_value=1, format="%d")

        distance = []
        for i in range(StudentPerformance.shape[0]):
            value = ((StudentPerformance['math score'][i] - mathscore) ** 2) + (
                        (StudentPerformance['reading score'][i] - readingscore) ** 2) + (
                                (StudentPerformance['writing score'][i] - writingscore) ** 2)
            value = np.sqrt(value)
            distance.append(value)

        m = 999
        iteration = 1
        lis1 = []
        lis2 = []
        lis3 = []
        lis4 = []
        while iteration <= k:
            mathscore = 0
            readingscore = 0
            writingscore = 0
            category = ""
            index = 0
            for i in range(StudentPerformance.shape[0]):
                if m > distance[i]:
                    m = distance[i]
                    mathscore = StudentPerformance['math score'][i]
                    readingscore = StudentPerformance['reading score'][i]
                    writingscore = StudentPerformance['writing score'][i]
                    category = StudentPerformance['category'][i]
                    index = i
            lis1.append(mathscore)
            lis2.append(readingscore)
            lis3.append(writingscore)
            lis4.append(category)
            distance[index] = 999
            m = 999
            iteration = iteration + 1

        if st.sidebar.button("Run Algorithm"):
            st.subheader("Table of k nearest points")
            st.write("Value of k:", k)
            data = {"mathscore": lis1, "readingscore": lis2, "writingscore": lis3, "category": lis4}
            data = pd.DataFrame(data)
            data.index = data.index + 1

            column = np.array(data['category'])
            value = {"A": 0, "B": 0, "C": 0, "D": 0}
            for i in range(data.shape[0]):
                if column[i] == "A":
                    value['A'] = value['A'] + 1
                elif column[i] == "B":
                    value['B'] = value['B'] + 1
                elif column[i] == "C":
                    value['C'] = value['C'] + 1
                elif column[i] == "D":
                    value['D'] = value['D'] + 1

            m = -999
            answer = ""
            for i in value:
                if m < value[i]:
                    m = value[i]
                    answer = i

            st.table(data)
            st.write("Output category is: ", answer)
elif Algorithm=="SupportVectorMachine":
    st.sidebar.markdown("# Support Vector Machine")
    X, Y = make_blobs(n_samples=100, centers=2, random_state=6)


    def graph(svm):
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=Y, s=30, cmap=plt.cm.Paired)

        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = svm.decision_function(xy).reshape(XX.shape)

        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], aplha=0.5, linestyles=['--', '-', '--'])
        ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none',
                   edgecolors='k')
        st.pyplot(plt.gcf())
        plt.clf()


    C = st.sidebar.number_input("Enter C", min_value=0.0, value=1.0)
    kernel = st.sidebar.selectbox(
        'Enter Kernel',
        ('rbf', 'linear', 'ploy', 'sigmoid', 'precomputed')
    )
    gamma = st.sidebar.selectbox(
        'Enter Gamma',
        ('scale', 'auto')
    )
    coef0 = st.sidebar.number_input("Enter coef0", min_value=0.0, value=0.0)
    shrinking = st.sidebar.selectbox(
        'Enter Shrinking',
        (True, False)
    )
    probability = st.sidebar.selectbox(
        'Enter Probability',
        (False, True)
    )
    verbose = st.sidebar.selectbox(
        'Select Verbose',
        (False, True)
    )
    max_iter = st.sidebar.number_input("Enter Max_iter", value=-1)
    decision_function_shape = st.sidebar.selectbox(
        ('Select Decision Function Shape'),
        ('ovr', 'ovo')
    )

    if st.sidebar.button("Run Algorithm"):
        svm = SVC(C=C, kernel=kernel, gamma=gamma, coef0=coef0, shrinking=shrinking, probability=probability,
                  verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape)
        svm.fit(X, Y)

        st.subheader("Parameters(with values)")
        st.write("C: ", C)
        st.write("Kernel: ", kernel)
        st.write("Gamma: ", gamma)
        st.write("Coef0: ", coef0)
        st.write("Shrinking: ", shrinking)
        st.write("Probability: ", probability)
        st.write("Verbose: ", verbose)
        st.write("Max_iter: ", max_iter)
        st.write("Decision Function Shape: ", decision_function_shape)
        graph(svm)
elif Algorithm == "Density-Based Spatial Clustering of Applications with Noise":
    st.sidebar.markdown("# DBSCAN Algorithm")
    Dataset = st.sidebar.selectbox(
        "Select Dataset",
        ('Select Dataset', 'Iris', 'Blobs', 'Moon', "Circle")
    )

    eps = st.sidebar.number_input("Enter Epsilon", min_value=0.0, value=0.5)
    min_samples = st.sidebar.number_input("Enter Min Samples", min_value=1, value=5)
    metric = st.sidebar.selectbox(
        'Select Metrics',
        ('euclidean', 'str','callable')
    )
    metric_params = st.sidebar.selectbox(
        'Select Metric Params',
        (None, 'dic')
    )
    algorithm = st.sidebar.selectbox(
        'Select Algorithm',
        ('auto', 'ball_tree', 'kd_tree', 'brute')
    )
    leaf_size = st.sidebar.number_input("Enter Leaf Size", min_value=1, format="%d", value=30)
    p = st.sidebar.number_input("Enter P", min_value=1, format="%d", value=2)
    n_jobs = st.sidebar.selectbox(
        'Enter n_jobs',
        (1, -1)
    )

    if Dataset == 'Iris':
        st.subheader("Parameters(with values")
        st.write("eps: ", eps)
        st.write("min_samples: ", min_samples)
        st.write("metric: ", metric)
        st.write("metric_params: ", metric_params)
        st.write("algorithm: ", algorithm)
        st.write("leaf_size: ", leaf_size)
        st.write("p: ", p)
        st.write("n_jobs: ", n_jobs)
        iris = load_iris()
        x = iris.data
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, metric_params=metric_params,
                        algorithm=algorithm, leaf_size=leaf_size, p=p, n_jobs=n_jobs)
        labels = dbscan.fit_predict(x)

        fig1, ax1 = plt.subplots()
        ax1.scatter(x[:, 0], x[:, 1], c=labels, s=30, cmap=plt.cm.Paired)

        for i in range(len(labels)):
            if labels[i] == -1:
                ax1.scatter(x[i, 0], x[i, 1], s=100, color='k', label='Noise', marker='x')

        fig2, ax2 = plt.subplots()
        ax2.scatter(x[:, 2], x[:, 3], c=labels, s=30, cmap=plt.cm.Paired)

        for i in range(len(labels)):
            if labels[i] == -1:
                ax2.scatter(x[i, 0], x[i, 1], s=100, color='k', label='Noise', marker='x')

        if st.sidebar.button("Run Algorithm"):
            st.subheader("Sepal(Length and Width)")
            st.pyplot(fig1)
            st.subheader("Petal(Length and Width)")
            st.pyplot(fig2)
    elif Dataset == 'Blobs':
        st.subheader("Parameters(with values")
        st.write("eps: ", eps)
        st.write("min_samples: ", min_samples)
        st.write("metric: ", metric)
        st.write("metric_params: ", metric_params)
        st.write("algorithm: ", algorithm)
        st.write("leaf_size: ", leaf_size)
        st.write("p: ", p)
        st.write("n_jobs: ", n_jobs)
        X, Y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, metric_params=metric_params,
                        algorithm=algorithm,
                        leaf_size=leaf_size, p=p, n_jobs=n_jobs)
        labels = dbscan.fit_predict(X)

        plt.figure(figsize=(8, 6))

        unique_labels = set(labels)
        colors = plt.cm.Paired(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            if label == -1:
                color = 'k'
            plt.scatter(X[labels == label, 0], X[labels == label, 1], s=30, color=color, label=f'Cluster {label}')

        if st.sidebar.button("Run Algorithm"):
            st.pyplot(plt.gcf())
            plt.clf()
    elif Dataset == 'Moon':
        st.subheader("Parameters(with values")
        st.write("eps: ", eps)
        st.write("min_samples: ", min_samples)
        st.write("metric: ", metric)
        st.write("metric_params: ", metric_params)
        st.write("algorithm: ", algorithm)
        st.write("leaf_size: ", leaf_size)
        st.write("p: ", p)
        st.write("n_jobs: ", n_jobs)
        X, Y = make_moons(n_samples=300, noise=0.05)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, metric_params=metric_params,
                        algorithm=algorithm,
                        leaf_size=leaf_size, p=p, n_jobs=n_jobs)
        labels = dbscan.fit_predict(X)

        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap=plt.cm.Paired)

        if st.sidebar.button("Run Algorithm"):
            st.pyplot(fig)
            plt.clf()
    elif Dataset == 'Circle':
        st.subheader("Parameters(with values")
        st.write("eps: ", eps)
        st.write("min_samples: ", min_samples)
        st.write("metric: ", metric)
        st.write("metric_params: ", metric_params)
        st.write("algorithm: ", algorithm)
        st.write("leaf_size: ", leaf_size)
        st.write("p: ", p)
        st.write("n_jobs: ", n_jobs)
        X, Y = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, metric_params=metric_params,
                        algorithm=algorithm,
                        leaf_size=leaf_size, p=p, n_jobs=n_jobs)
        labels = dbscan.fit_predict(X)

        plt.figure(figsize=(8, 6))

        unique_labels = set(labels)
        colors = plt.cm.Paired(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            if label == -1:
                color = 'k'
            plt.scatter(X[labels == label, 0], X[labels == label, 1], s=30, color=color, label=f'Cluster {label}')

        if st.sidebar.button("Run Algorithm"):
            st.pyplot(plt.gcf())
            plt.clf()