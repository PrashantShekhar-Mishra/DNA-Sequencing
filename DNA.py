import time
from tkinter.tix import COLUMN
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from multiprocessing import Process
import gentic
from antcolony import AntColonyOptimizer


#Removing crosstab warning
st.set_option('deprecation.showPyplotGlobalUse', False)
multiplegraph=[]
# function to convert sequence strings into k-mer words, default size = 4 (hexamer words)
def getKmers(sequence, size=8):
    return list(sequence[x:x+size].lower() for x in range(len(sequence) - size + 1))



#accuracy precision recall f1 
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1




#data extraction
# The n-gram size of 4 was previously determined by testing


def dataext(data_file):
    df=pd.read_table(data_file)
    st.dataframe(df)
    df['words'] = df.apply(lambda x: getKmers(x['sequence']), axis=1)
    df = df.drop('sequence', axis=1)
    return df




# confusion_matrix
def confusion_mat(model,X_test,y_test):
    plot_confusion_matrix(model, X_test, y_test)
    st.pyplot()




#piechartchart
def pie_chart(df):
    st.write("_________________________________________________________________________________________________________________________________")
    bar_data=list(df['class'].value_counts().sort_index())
    label=['Class -> 0-G protein coupled receptors ','Class -> 1-Tyrosine kinase ','Class -> 2-Tyrosine phosphatase','Class -> 3- Synthetase','Class -> 4- Synthase ','Class -> 5-lon channel ','Class -> 6-Transcription factor']
    fig = go.Figure(
    go.Pie(
    labels = label,
    values = bar_data,
    hoverinfo = "label+percent",
    textinfo = "value"
    ))

    st.subheader("Pie chart")
    st.plotly_chart(fig)


def _model_(model,X_train, X_test, y_train, y_test):
    model.fit(X_train,y_train)
    model.fit(X_train, y_train)
    st.subheader("Confusion matrix\n")
    confusion_mat(model,X_test,y_test)
    y_predicted = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    st.subheader("Accuracy = %.3f   ,   Precision = %.3f    ,    recall = %.3f    ,    f1 = %.3f" % (accuracy, precision, recall, f1))
    multiplegraph.append(list([accuracy,precision,recall,f1]))




#Classifiers
#svm
def svm_class(X_train, X_test, y_train, y_test,exe):
    st.subheader("SVM") 
    model=SVC(kernel='linear')
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    if(exe==1):
        st.subheader("Predicted sequence ")
        st.write(*y_pred)
        st.subheader("Accuracy = %.3f , Precision = %.3f , recall = %.3f , f1 = %.3f" % (accuracy, precision, recall, f1))  
    if(exe==0):
        st.subheader("Confusion matrix\n")
        confusion_mat(model,X_test,y_test)
        st.write("")
        st.subheader("Accuracy = %.3f , Precision = %.3f , recall = %.3f , f1 = %.3f" % (accuracy, precision, recall, f1))
    multiplegraph.append(list([accuracy,precision,recall,f1]))
    return f1

#Random Forest
def random_forest(X_train, X_test, y_train, y_test,exe):  
    st.write("_________________________________________________________________________________________________________________________________")
    st.subheader("Random Forest") 
    model=RandomForestClassifier(max_depth=20, n_estimators=20, max_features=20)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    if(exe==1):
        st.subheader("Predicted sequence ")
        st.write(*y_pred)
        st.subheader("Accuracy = %.3f , Precision = %.3f , recall = %.3f , f1 = %.3f" % (accuracy, precision, recall, f1))
    if(exe==0):
        st.subheader("Confusion matrix\n")
        confusion_mat(model,X_test,y_test)
        st.write("")
        st.subheader("Accuracy = %.3f , Precision = %.3f , recall = %.3f , f1 = %.3f" % (accuracy, precision, recall, f1))
    multiplegraph.append(list([accuracy,precision,recall,f1]))
    return f1

#Knn
def knn_class(X_train, X_test, y_train, y_test,exe):
    st.write("_________________________________________________________________________________________________________________________________")
    st.subheader("KNN")   
    model=KNeighborsClassifier(n_neighbors=6)
    model.fit(X_train, y_train)
    st.write("")
    y_pred = model.predict(X_test)
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    if(exe==1):
        st.subheader("Predicted sequence ")
        st.write(*y_pred)
        st.subheader("Accuracy = %.3f , Precision = %.3f , recall = %.3f , f1 = %.3f" % (accuracy, precision, recall, f1))   
    if(exe==0):
        st.subheader("Confusion matrix\n")
        confusion_mat(model,X_test,y_test)
        st.subheader("Accuracy = %.3f , Precision = %.3f , recall = %.3f , f1 = %.3f" % (accuracy, precision, recall, f1))
    multiplegraph.append(list([accuracy,precision,recall,f1]))
    return f1

#Decison tree
def Deci_tree(X_train, X_test, y_train, y_test,exe):
    st.write("_________________________________________________________________________________________________________________________________")
    st.subheader("Decision Tree") 
    model=DecisionTreeClassifier(max_depth=20)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    if(exe==1):
        st.subheader("Predicted sequence ")
        st.write(*y_pred)
        st.subheader("Accuracy = %.3f , Precision = %.3f , recall = %.3f , f1 = %.3f" % (accuracy, precision, recall, f1))
    if(exe==0):
        st.subheader("Confusion matrix\n")
        st.write("")
        confusion_mat(model,X_test,y_test)
        st.subheader("Accuracy = %.3f , Precision = %.3f , recall = %.3f , f1 = %.3f" % (accuracy, precision, recall, f1))
    multiplegraph.append(list([accuracy,precision,recall,f1]))
    return f1


#Main 
if __name__ == '__main__':
    st.title("DNA SEQUENCING")
    st.write("_________________________________________________________________________________________________________________________________")
    data_file = st.file_uploader("Upload CSV",type=["csv","txt"])
    if data_file is not None:
        file_details = {"filename":data_file.name, "filetype":data_file.type, "filesize":data_file.size}
        add_selectbox = st.sidebar.selectbox("Which operation do you want to follow?", (" ","Data Extraction","DNA Sequencing","DNA Sequencing using Genetic Algorithm","Optimize and Paralleize the above two processes."))
        if(add_selectbox==" "):
                st.header("Please select an option...")
        elif(add_selectbox=="Data Extraction"):
            df=dataext(data_file)
            st.write("_________________________________________________________________________________________________________________________________")
            st.write("")
            st.subheader(" K-mers")
            dft= list(df['words'])
            for item in range(len(df)):
                dft[item] = ' '.join(dft[item])
            y_data = df.iloc[:, 0].values
            st.write(" ") #newline
            #data 
            st.write(df)
            st.write("") #newline
    
            #piechart
            pie_chart(df) 


            st.write("") #newline  
            st.write("") #newline
            multiplegraph=[]


        elif(add_selectbox=="DNA Sequencing"):
            df=dataext(data_file)
            st.write("_________________________________________________________________________________________________________________________________")
            st.write("")
            st.subheader(" K-mers")
            dft= list(df['words'])
            for item in range(len(df)):
                dft[item] = ' '.join(dft[item])
            y_data = df.iloc[:, 0].values
            st.write(" ") #newline
            #data 
            st.write(df)
            st.write("") #newline
    
            #piechart
            pie_chart(df)
            # Creating the Bag of Words model using CountVectorizer()
            # This is equivalent to k-mer counting
            cv = CountVectorizer(ngram_range=(4,4))
            X = cv.fit_transform(dft)

            #training dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size = 0.2, random_state=20)
            #Classifiers
            modl=["SVM ","Random Forest","KNN","Decision Tree"]
            s=time.time()
            f1=[]
            f1.append(svm_class(X_train, X_test, y_train, y_test,1))
            f1.append(random_forest(X_train, X_test, y_train, y_test,1))
            f1.append(knn_class(X_train, X_test, y_train, y_test,1))
            f1.append(Deci_tree(X_train, X_test, y_train, y_test,1))
            mx=f1.index(max(f1))
            st.write("")
            val="{:.3f}".format(max(f1))      #Formating upto 3 precison
            y=["Accuracy","Precison","Recall","f1"]
            st.write("_________________________________________________________________________________________________________________________________")
            i=0
            st.header("RESULT :-")
            fig1 = plt.figure(figsize = (9,3))
            for x in multiplegraph:
                plt.plot(y,x,label=modl[i])
                i+=1
            plt.legend()
            st.pyplot(fig1)
            st.subheader(f"Best classifier is : {modl[mx]} with F1 score : {val}")  #final result
            multiplegraph=[]

        elif(add_selectbox=="DNA Sequencing using Genetic Algorithm"):
            df=pd.read_table(data_file)
            st.dataframe(df)
            df['words'] = df.apply(lambda x: getKmers(x['sequence']), axis=1)
            df = df.drop('sequence', axis=1)
            st.write("_________________________________________________________________________________________________________________________________")
            st.write("")
            st.subheader("K-mers")
            dft= list(df['words'])
            for item in range(len(df)):
                dft[item] = ' '.join(dft[item])
            y_data = df.iloc[:, 0].values

            st.write(" ") #newline
            #data 
            st.write(df)
            st.write("") #newline

            pie_chart(df) 

            st.write("") #newline  
            st.write("") #newline

 
            # Creating the Bag of Words model using CountVectorizer()
            # This is equivalent to k-mer counting
            cv = CountVectorizer(ngram_range=(4,4))
            X = cv.fit_transform(dft)

            #training dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size = 0.2, random_state=20)
            start1=time.time()
            st.subheader("GENETIC ALGORITHM")
            model=gentic.Genetic()
            _model_(model,X_train, X_test, y_train, y_test)
            end1=time.time()
            t=end1-start1
            y=["Accuracy","Precison","Recall","f1"]
            st.write("_________________________________________________________________________________________________________________________________")
            i=0
            modl=["GENETIC ALGORITHM"]
            st.header("RESULT :-")
            fig1 = plt.figure(figsize = (9,3))
            for x in multiplegraph:
                plt.plot(y,x,label=modl[i])
                i+=1
            plt.legend()
            st.pyplot(fig1)
            st.subheader(f"Time required for genetic algorithm :- {t} Sec")
            st.write("")
            y_pred = model.predict(X_test)
            prob=pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted'))
            problem=[]
            st.write("_____________________________________________________________________________________________________________________________")
            st.subheader("Ant colony distance matrix")
            st.write(prob)
            modl.append("Ant colony")
            for i in range(1,len(prob)):
                x=[]
                for j in range(1,len(prob)):
                    x.append(prob[i][j])
                problem.append(x)
            multiplegraph.append(AntColonyOptimizer._model_(model,X_train,X_test,y_train,y_test))
            fig1 = plt.figure(figsize = (9,3))
            i=0
            for x in multiplegraph:
                plt.plot(y,x,label=modl[i])
                i+=1
            plt.legend()
            st.pyplot(fig1)
            st.subheader(f"Time required Antcolony  :- {AntColonyOptimizer.optimize(t,problem)} Sec")
            multiplegraph=[]



        elif(add_selectbox=="Optimize and Paralleize the above two processes."):
            multiplegraph=[]
            df=dataext(data_file)
            st.write("_________________________________________________________________________________________________________________________________")
            st.write("")
            st.subheader(" K-mers")
            dft= list(df['words'])
            for item in range(len(df)):
                dft[item] = ' '.join(dft[item])
            y_data = df.iloc[:, 0].values
            st.write(" ") #newline
            #data 
            st.write(df)
            st.write("") #newline
    
            #piechart
            pie_chart(df)
            # Creating the Bag of Words model using CountVectorizer()
            # This is equivalent to k-mer counting
            cv = CountVectorizer(ngram_range=(4,4))
            X = cv.fit_transform(dft)

            #training dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size = 0.2, random_state=20)
            #Classifiers
            modl=["SVM ","Random Forest","KNN","Decision Tree"]
            #parallel processing
            #f1_list
            t=list()
            st.write("_________________________________________________________________________________________________________________________________")
            st.header("WITHOUT PARALLEL PROCESSING")
            s=time.time()
            svm_class(X_train, X_test, y_train, y_test,1)
            e1=time.time()
            random_forest(X_train, X_test, y_train, y_test,1)
            e2=time.time()
            knn_class(X_train, X_test, y_train, y_test,1)
            e3=time.time()
            Deci_tree(X_train, X_test, y_train, y_test,1)
            e4=time.time()
            t1=e1-s
            t2=e2-s
            t3=e3-s
            t4=e4-s
            t.append(list([t1,t2,t3,t4]))
            f1=[]
            st.write("_________________________________________________________________________________________________________________________________")
            st.header("WITH PARALLEL PROCESSING")
            s=time.time()
            p1=Process(f1.append(svm_class(X_train, X_test, y_train, y_test,0)))
            p1.start()
            p2=Process(f1.append(random_forest(X_train, X_test, y_train, y_test,0)))
            p2.start()
            p3=Process(f1.append(knn_class(X_train, X_test, y_train, y_test,0)))
            p3.start()
            p4=Process(f1.append(Deci_tree(X_train, X_test, y_train, y_test,0)))
            p4.start()
            p1.join()
            e1=time.time()
            p2.join()
            e2=time.time()
            p3.join()
            e3=time.time()
            p4.join()
            e4=time.time()
            t1=e1-s
            t2=e2-s
            t3=e3-s
            t4=e4-s
            t.append(list([t1,t2,t3,t4]))
            t1="{:.3f}".format(t[1][3])
            t2="{:.3f}".format(t[0][3])
            #finding Max out of f1
            mx=f1.index(max(f1))
            st.write("")
            val="{:.3f}".format(max(f1))      #Formating upto 3 precison
            y=["Accuracy","Precison","Recall","f1"]
            st.write("_________________________________________________________________________________________________________________________________")
            i=0
            st.header("RESULT :-")
            fig1 = plt.figure(figsize = (9,3))
            for x in multiplegraph:
                plt.plot(y,x,label=modl[i])
                i+=1
                if(i==4):
                    break
            plt.legend()
            st.pyplot(fig1)
            y=[]
            lbl=["With parallel Processing","Without parallel processing"]
            st.subheader("Time analysis after each Process :-")
            fig1 = plt.figure(figsize = (9,3))
            st.subheader("Value Comparison")
            fig1 = plt.figure(figsize = (9,3))
            i=0
            for x in t:
                plt.plot(modl,x,label=lbl[i])
                i+=1
            plt.legend()
            st.pyplot(fig1)
            st.subheader("Result comparison")
            fig2 = plt.figure(figsize = (9,3))
            plt.hist([t[0],t[1]],label=["With parallel processing","Without parallel processing"])
            plt.legend(loc='upper right')   
            plt.legend()
            st.pyplot(fig2)
            st.subheader(f"Best classifier is : {modl[mx]} with F1 score : {val}")  #final result
            st.subheader(f"Time without Parallel processing :- {t1} Sec")      
            st.subheader(f"Time with Parallel processing :- {t2} Sec")
            st.write("_________________________________________________________________________________________________________________________________")
            multiplegraph=[]