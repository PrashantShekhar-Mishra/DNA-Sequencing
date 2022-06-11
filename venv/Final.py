import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from multiprocessing import Process


#page layout
st.set_page_config(layout="wide")

#Removing crosstab warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# function to convert sequence strings into k-mer words, default size = 4 (hexamer words)
def getKmers(sequence, size=4):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]



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
    st.subheader("Heat Map")
    df['Length'] = df.apply(lambda x: len(getKmers(x['sequence'])), axis=1)
    df['words'] = df.apply(lambda x: getKmers(x['sequence']), axis=1)
    fig=plt.figure(figsize=(6,6))
    cor = df.corr()
    sns.heatmap(cor, cmap=plt.cm.Reds)
    st.write(fig)
    df = df.drop('Length', axis=1)
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
    label=['Class -> 0','Class -> 1','Class -> 2','Class -> 3','Class -> 4','Class -> 5','Class -> 6']
    fig = go.Figure(
    go.Pie(
    labels = label,
    values = bar_data,
    hoverinfo = "label+percent",
    textinfo = "value"
    ))

    st.subheader("Pie chart")
    st.plotly_chart(fig)



multiplegraph=[]

#Classifiers
#svm
def svm_class(X_train, X_test, y_train, y_test,exe):
    st.subheader("SVM") 
    if(exe==0):
        model=SVC(kernel='linear')
    else:
        model=SVC(kernel='rbf')
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    if(exe==1):
        st.subheader("Accuracy = %.3f   ,   Precision = %.3f    ,    recall = %.3f    ,    f1 = %.3f" % (accuracy, precision, recall, f1))
    if(exe==0):
        st.subheader("Confusion matrix\n")
        confusion_mat(model,X_test,y_test)
        st.write("")
        st.subheader("Accuracy = %.3f   ,   Precision = %.3f    ,    recall = %.3f    ,    f1 = %.3f" % (accuracy, precision, recall, f1))
        multiplegraph.append(list([accuracy,precision,recall,f1]))
    return f1

#Random Forest
def random_forest(X_train, X_test, y_train, y_test,exe):  
    st.write("_________________________________________________________________________________________________________________________________")
    st.subheader("Random Forest")
    if(exe==0):
        model=RandomForestClassifier(max_depth=30, n_estimators=30, max_features=30)
    else:
        model=RandomForestClassifier(max_depth=20, n_estimators=20, max_features=20)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    if(exe==1):
        st.subheader("Accuracy = %.3f   ,   Precision = %.3f    ,    recall = %.3f    ,    f1 = %.3f" % (accuracy, precision, recall, f1))
    if(exe==0):
        st.subheader("Confusion matrix\n")
        confusion_mat(model,X_test,y_test)
        st.write("")
        st.subheader("Accuracy = %.3f   ,   Precision = %.3f    ,    recall = %.3f    ,    f1 = %.3f" % (accuracy, precision, recall, f1))
        multiplegraph.append(list([accuracy,precision,recall,f1]))
    return f1

#Knn
def knn_class(X_train, X_test, y_train, y_test,exe):
    st.write("_________________________________________________________________________________________________________________________________")
    st.subheader("KNN")
    if(exe==0):
        model=KNeighborsClassifier(n_neighbors=3)
    else:
        model=KNeighborsClassifier(n_neighbors=7)
    
    model.fit(X_train, y_train)
    st.write("")
    y_pred = model.predict(X_test)
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    if(exe==1):
        st.subheader("Accuracy = %.3f   ,   Precision = %.3f    ,    recall = %.3f    ,    f1 = %.3f" % (accuracy, precision, recall, f1))
    if(exe==0):
        st.subheader("Confusion matrix\n")
        confusion_mat(model,X_test,y_test)
        st.subheader("Accuracy = %.3f   ,   Precision = %.3f    ,    recall = %.3f    ,    f1 = %.3f" % (accuracy, precision, recall, f1))
        multiplegraph.append(list([accuracy,precision,recall,f1]))
    return f1

#Decison tree
def Deci_tree(X_train, X_test, y_train, y_test,exe):
    st.write("_________________________________________________________________________________________________________________________________")
    st.subheader("Decision Tree") 
    if(exe==0):
        model=DecisionTreeClassifier(max_depth=50)
    else:
        model=DecisionTreeClassifier(max_depth=20)
    
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    if(exe==1):
        st.subheader("Accuracy = %.3f   ,   Precision = %.3f    ,    recall = %.3f    ,    f1 = %.3f" % (accuracy, precision, recall, f1))
    if(exe==0): 
        st.subheader("Confusion matrix\n")
        st.write("")
        confusion_mat(model,X_test,y_test)
        st.subheader("Accuracy = %.3f   ,   Precision = %.3f    ,    recall = %.3f    ,    f1 = %.3f" % (accuracy, precision, recall, f1))
        multiplegraph.append(list([accuracy,precision,recall,f1]))
    return f1


#Main 
if __name__ == '__main__':
    st.title("DNA SEQUENCING")
    st.write("_________________________________________________________________________________________________________________________________")
    data_file = st.file_uploader("Upload CSV",type=["csv","txt"])
    if data_file is not None:
        file_details = {"filename":data_file.name, "filetype":data_file.type, "filesize":data_file.size}
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

 
        # Creating the Bag of Words model using CountVectorizer()
        # This is equivalent to k-mer counting
        cv = CountVectorizer(ngram_range=(4,4))
        X = cv.fit_transform(dft)

        #training dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size = 0.2, random_state=20)
        start1=time.time()
        #Classifiers
        modl=["SVM ","Random Forest","KNN","Decision Tree"]
        #parallel processing
        #f1_list
        t=[]
        st.write("_________________________________________________________________________________________________________________________________")
        st.header("WITHOUT PARALLEL PROCESSING")
        svm_class(X_train, X_test, y_train, y_test,1)
        random_forest(X_train, X_test, y_train, y_test,1)
        knn_class(X_train, X_test, y_train, y_test,1)
        Deci_tree(X_train, X_test, y_train, y_test,1)
        end1=time.time()
        total_time1= end1-start1
        t.append(total_time1)
        f1=[]
        st.write("_________________________________________________________________________________________________________________________________")
        st.header("WITH PARALLEL PROCESSING")
        start1=time.time()
        p1=Process(f1.append(svm_class(X_train, X_test, y_train, y_test,0)))
        p1.start()
        p2=Process(f1.append(random_forest(X_train, X_test, y_train, y_test,0)))
        p2.start()
        p3=Process(f1.append(knn_class(X_train, X_test, y_train, y_test,0)))
        p3.start()
        p4=Process(f1.append(Deci_tree(X_train, X_test, y_train, y_test,0)))
        p4.start()
        p1.join()
        p2.join()
        p3.join()
        p4.join()
        end1=time.time()
        total_time1= end1-start1
        t.append(total_time1)
        t[0],t[1]=t[1],t[0]
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
        plt.legend()
        st.pyplot(fig1)
        st.subheader(f"Best classifier is : {modl[mx]} with F1 score : {val}")  #final result
        st.subheader(f"Time without Parallel processing :- {t[1]}")      
        st.subheader(f"Time with Parallel processing :- {t[0]}")
        st.write("_________________________________________________________________________________________________________________________________")
