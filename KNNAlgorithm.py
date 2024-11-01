import streamlit as st
import numpy as np
import pandas as pd
import pickle

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