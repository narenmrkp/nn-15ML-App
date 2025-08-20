# ml_projects.py
import os
HF_API_KEY = os.getenv("HF_API_KEY")

# Example: using HF dataset loading
from datasets import load_dataset

# dataset = load_dataset("your_dataset_name", use_auth_token=HF_API_KEY)

from dotenv import load_dotenv
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
if HF_API_KEY:
    try:
        from huggingface_hub import login
        login(HF_API_KEY)
    except Exception:
        # best-effort; if login not possible, continue (public datasets often work)
        pass

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# Utility: safe HF loader
def safe_load_hf(hf_id, split="train"):
    """
    Try load_dataset(hf_id). Return pd.DataFrame or raise.
    Use small subset where possible: many HF datasets accept split like 'train[:2000]'.
    """
    try:
        ds = load_dataset(hf_id, split=split)
        df = pd.DataFrame(ds)
        return df
    except Exception as e:
        raise

# Utility: robust train_test_split with fallback to non-stratified when classes small
def robust_split(X, y, test_size=0.2, stratify=True):
    try:
        if stratify:
            counts = pd.Series(y).value_counts()
            if counts.min() < 2:
                return train_test_split(X, y, test_size=test_size, random_state=42)
            return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        else:
            return train_test_split(X, y, test_size=test_size, random_state=42)
    except Exception:
        return train_test_split(X, y, test_size=test_size, random_state=42)

# -------------------------
# 01 - Random Forest Fraud
# -------------------------
@st.cache_data(show_spinner=False)
def _train_fraud():
    # Try HF mirrored datasets, then fallback to sklearn/synthetic
    df = None
    for candidate in [
        "jiaqiangz/creditcard-fraud",  # possible mirror
        "fraud-detection/creditcard",  # alternate
        "mlg-ulb/creditcard-fraud"     # commonly used id (may not exist on HF)
    ]:
        try:
            df = safe_load_hf(candidate, split="train[:20000]")
            break
        except Exception:
            df = None
    if df is None:
        # fallback synthetic
        rng = np.random.RandomState(42)
        n = 5000
        Xsyn = rng.normal(size=(n, 10))
        amount = rng.exponential(scale=100, size=(n,1))
        Xdf = pd.DataFrame(Xsyn, columns=[f"V{i}" for i in range(1,11)])
        df = pd.concat([Xdf, pd.DataFrame(amount, columns=["Amount"])], axis=1)
        df["Class"] = (rng.rand(n) < 0.02).astype(int)

    # try to standardize label column name
    if "Class" not in df.columns:
        for c in ["class","isFraud","fraud","label"]:
            if c in df.columns:
                df = df.rename(columns={c:"Class"})
                break
    if "Class" not in df.columns:
        df["Class"] = (np.random.rand(len(df)) < 0.02).astype(int)

    X = df.select_dtypes(include=[np.number]).drop(columns=["Class"], errors="ignore").fillna(0)
    y = df["Class"]
    X_train, X_test, y_train, y_test = robust_split(X, y)
    scaler = StandardScaler().fit(X_train)
    model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    model.fit(scaler.transform(X_train), y_train)
    return model, scaler, X_test, y_test

def proj_01_rf_fraud():
    st.header("01 • Random Forest — Credit Card Fraud Detection")
    st.write("Dataset: Hugging Face (preferred) — fallback to synthetic if HF missing.")
    model, scaler, X_test, y_test = _train_fraud()
    preds = model.predict(scaler.transform(X_test))
    st.subheader("Evaluation")
    st.write("Accuracy:", accuracy_score(y_test, preds))
    st.text(classification_report(y_test, preds, zero_division=0))
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", ax=ax, cmap="Reds")
    st.pyplot(fig)

    st.subheader("Single-sample prediction (comma-separated numeric values)")
    sample = st.text_input("Enter values", value=",".join(["0"] * X_test.shape[1]))
    if st.button("Predict Fraud (RF)"):
        try:
            arr = np.array([float(x) for x in sample.split(",")])
            if arr.shape[0] != X_test.shape[1]:
                st.error(f"Expecting {X_test.shape[1]} values (got {arr.shape[0]})")
            else:
                p = model.predict(scaler.transform([arr]))[0]
                prob = model.predict_proba(scaler.transform([arr]))[0][1] if hasattr(model, "predict_proba") else None
                st.write("Prediction:", "Fraud 🚨" if p==1 else "Legit ✅")
                if prob is not None:
                    st.write("Prob(Fraud):", round(float(prob),3))
        except Exception as e:
            st.error("Invalid input: " + str(e))

# -------------------------
# 02 - Decision Tree Diabetes
# -------------------------
@st.cache_data(show_spinner=False)
def _train_diabetes():
    df = None
    try:
        df = safe_load_hf("jiaqiangz/pima-indians-diabetes-database", split="train[:2000]")
    except Exception:
        try:
            df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
        except Exception:
            # sklearn fallback
            from sklearn.datasets import load_diabetes
            d = load_diabetes(as_frame=True)
            df = d.frame
            df["Outcome"] = (d.target > d.target.mean()).astype(int)
    if "Outcome" not in df.columns:
        for cand in ["Outcome","outcome","target","Target"]:
            if cand in df.columns:
                df = df.rename(columns={cand:"Outcome"})
                break
    if "Outcome" not in df.columns:
        df["Outcome"] = (np.random.rand(len(df))>0.5).astype(int)
    X = df.drop(columns=["Outcome"]).select_dtypes(include=[np.number]).fillna(0)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = robust_split(X, y)
    scaler = StandardScaler().fit(X_train)
    model = DecisionTreeClassifier(max_depth=6, random_state=42)
    model.fit(scaler.transform(X_train), y_train)
    return model, scaler, X_test, y_test

def proj_02_dt_diabetes():
    st.header("02 • Decision Tree — Diabetes Prediction")
    model, scaler, X_test, y_test = _train_diabetes()
    Xte = scaler.transform(X_test)
    preds = model.predict(Xte)
    st.write("Accuracy:", accuracy_score(y_test, preds))
    st.text(classification_report(y_test, preds, zero_division=0))
    if st.checkbox("Show tree (top levels)"):
        fig = plt.figure(figsize=(10,4))
        plot_tree(model, max_depth=3, filled=True)
        st.pyplot(fig)

    st.subheader("Single sample (numeric features)")
    defaults = list(X_test.iloc[0])
    inputs = [st.number_input(f"feat_{i+1}", value=float(defaults[i]), key=f"dia_{i}") for i in range(X_test.shape[1])]
    if st.button("Predict (DT)"):
        val = np.array(inputs).reshape(1,-1)
        p = model.predict(scaler.transform(val))[0]
        st.write("Prediction:", "Diabetic" if p==1 else "Not Diabetic")

# -------------------------
# 03 - SVM Cancer
# -------------------------
@st.cache_data(show_spinner=False)
def _train_svm_cancer():
    try:
        df = safe_load_hf("breast-cancer-wisconsin", split="train[:1000]")
    except Exception:
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer(as_frame=True)
        df = d.frame
        df['target'] = d.target
    if 'target' in df.columns:
        X = df.select_dtypes(include=[np.number]).drop(columns=['target'], errors='ignore').fillna(0)
        y = df['target']
    else:
        # fallback handle
        X = df.select_dtypes(include=[np.number]).fillna(0)
        y = (np.random.rand(len(X)) > 0.5).astype(int)
    X_train, X_test, y_train, y_test = robust_split(X, y)
    scaler = StandardScaler().fit(X_train)
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(scaler.transform(X_train), y_train)
    return model, scaler, X_test, y_test

def proj_03_svm_cancer():
    st.header("03 • SVM — Cancer Detection")
    model, scaler, X_test, y_test = _train_svm_cancer()
    preds = model.predict(scaler.transform(X_test))
    st.write("Accuracy:", accuracy_score(y_test, preds))
    st.text(classification_report(y_test, preds, zero_division=0))
    defaults = list(X_test.iloc[0])
    inputs = [st.number_input(f"feat_{i+1}", value=float(defaults[i]), key=f"svm_{i}") for i in range(X_test.shape[1])]
    if st.button("Predict (SVM)"):
        p = model.predict(scaler.transform([inputs]))[0]
        st.write("Prediction:", "Malignant" if p==1 else "Benign")

# -------------------------
# 04 - Logistic Regression Heart
# -------------------------
@st.cache_data(show_spinner=False)
def _train_logreg_heart():
    try:
        df = safe_load_hf("rahulrajpl/heart", split="train[:2000]")
    except Exception:
        try:
            df = pd.read_csv("https://raw.githubusercontent.com/rahulrajpl/data/main/heart.csv")
        except Exception:
            n=300; rng=np.random.RandomState(42)
            df = pd.DataFrame({"age":rng.randint(29,78,n),"trestbps":rng.randint(94,180,n),
                               "chol":rng.randint(126,564,n),"thalach":rng.randint(71,202,n),
                               "target":rng.randint(0,2,n)})
    if 'target' not in df.columns:
        for cand in ['target','Target','TARGET']:
            if cand in df.columns:
                df = df.rename(columns={cand:'target'})
                break
    if 'target' not in df.columns:
        df['target'] = (np.random.rand(len(df))>0.5).astype(int)
    X = df.select_dtypes(include=[np.number]).drop(columns=['target'], errors='ignore').fillna(0)
    y = df['target']
    X_train, X_test, y_train, y_test = robust_split(X,y)
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)
    return model, scaler, X_test, y_test

def proj_04_logreg_heart():
    st.header("04 • Logistic Regression — Heart Disease")
    model, scaler, X_test, y_test = _train_logreg_heart()
    preds = model.predict(scaler.transform(X_test))
    st.write("Accuracy:", accuracy_score(y_test, preds))
    st.text(classification_report(y_test, preds, zero_division=0))
    # single sample
    defaults = list(X_test.iloc[0])
    inputs = [st.number_input(f"feat_{i+1}", value=float(defaults[i]), key=f"heart_{i}") for i in range(X_test.shape[1])]
    if st.button("Predict (LogReg)"):
        vals = np.array(inputs).reshape(1,-1)
        p = model.predict(scaler.transform(vals))[0]
        prob = model.predict_proba(scaler.transform(vals))[0][1]
        st.write("Prediction:", "At Risk" if p==1 else "Low Risk")
        st.write("Probability:", round(float(prob),3))

# -------------------------
# 05 - KNN Iris
# -------------------------
@st.cache_data(show_spinner=False)
def _train_knn_iris():
    try:
        ds = load_dataset("iris")
        df = pd.DataFrame(ds)
    except Exception:
        from sklearn.datasets import load_iris
        d = load_iris(as_frame=True)
        df = d.frame
        df['target'] = d.target
    if 'target' in df.columns:
        X = df.select_dtypes(include=[np.number]).drop(columns=['target'], errors='ignore').fillna(0)
        y = df['target']
    else:
        X = df.select_dtypes(include=[np.number]).fillna(0)
        y = (np.random.rand(len(X))>0.5).astype(int)
    scaler = StandardScaler().fit(X)
    X_train, X_test, y_train, y_test = robust_split(X,y)
    model = KNeighborsClassifier(n_neighbors=5).fit(scaler.transform(X_train), y_train)
    return model, scaler, X_test, y_test, X.columns if hasattr(X,"columns") else [f"f{i}" for i in range(X.shape[1])]

def proj_05_knn_iris():
    st.header("05 • KNN — Iris Classification")
    model, scaler, X_test, y_test, feat_names = _train_knn_iris()
    preds = model.predict(scaler.transform(X_test))
    st.write("Accuracy:", accuracy_score(y_test, preds))
    st.text(classification_report(y_test, preds, zero_division=0))
    defaults = list(X_test.iloc[0])
    inputs = [st.number_input(f"{feat_names[i]}", value=float(defaults[i]), key=f"iris_{i}") for i in range(X_test.shape[1])]
    if st.button("Predict (KNN)"):
        p = model.predict(scaler.transform([inputs]))[0]
        st.write("Prediction class id:", int(p))

# -------------------------
# 06 - NB Spam
# -------------------------
@st.cache_data(show_spinner=False)
def _train_nb_spam():
    df = None
    try:
        df = safe_load_hf("sms_spam_collection/smsspamcollection", split="train[:2000]")
    except Exception:
        try:
            df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep="\t", header=None, names=["label","message"])
        except Exception:
            df = pd.DataFrame({"label":["ham","spam","ham"], "message":["hello","win cash","see you"]})
    df = df.rename(columns={df.columns[0]:"label", df.columns[1]:"message"})
    df['y'] = (df['label'].astype(str).str.lower()=="spam").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['y'], test_size=0.2, random_state=42)
    vec = TfidfVectorizer(stop_words='english', max_features=2000)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    model = MultinomialNB().fit(Xtr, y_train)
    return model, vec, X_test, y_test

def proj_06_nb_spam():
    st.header("06 • Naive Bayes — SMS Spam Detection")
    model, vec, X_test, y_test = _train_nb_spam()
    preds = model.predict(vec.transform(X_test))
    st.write("Accuracy:", accuracy_score(y_test, preds))
    st.text(classification_report(y_test, preds, zero_division=0))
    txt = st.text_area("Enter SMS", value="Free offer, claim now")
    if st.button("Predict (NB)"):
        p = model.predict(vec.transform([txt]))[0]
        st.write("Prediction:", "Spam" if p==1 else "Not Spam")

# -------------------------
# 07 - Linear Regression Student
# -------------------------
@st.cache_data(show_spinner=False)
def _train_lr_student():
    try:
        df = safe_load_hf("student-performance/student-mat", split="train[:2000]")
    except Exception:
        np.random.seed(42)
        N=300
        hours=np.random.uniform(0,10,N)
        att=np.random.uniform(40,100,N)
        score=5*hours+0.3*att+np.random.normal(0,5,N)
        df=pd.DataFrame({"hours":hours,"attendance":att,"score":score})
    X = df[["hours","attendance"]] if "hours" in df.columns else df.iloc[:,:2]
    y = df["score"] if "score" in df.columns else df.iloc[:,2]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = LinearRegression().fit(X_train,y_train)
    return model, X_test, y_test

def proj_07_lr_student():
    st.header("07 • Linear Regression — Student Score")
    model, X_test, y_test = _train_lr_student()
    preds = model.predict(X_test)
    st.write("R2:", r2_score(y_test, preds))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
    hours = st.number_input("Study hours", value=5.0)
    attendance = st.number_input("Attendance (%)", value=80.0)
    if st.button("Predict (LR)"):
        pred = model.predict([[hours,attendance]])[0]
        st.write("Predicted Score:", round(float(pred),2))

# -------------------------
# 08 - RF Regression House Price
# -------------------------
@st.cache_data(show_spinner=False)
def _train_rfr_house():
    try:
        df = safe_load_hf("house-prices/house_prices", split="train[:2000]")
    except Exception:
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing(as_frame=True)
        df = d.frame
    # Determine target column
    if "MedHouseVal" in df.columns:
        target = "MedHouseVal"
    elif "MedInc" in df.columns and "target" in df.columns:
        target = "target"
    else:
        target = df.columns[-1]
    X = df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore').fillna(0)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    scaler = StandardScaler().fit(X_train)
    model = RandomForestRegressor(n_estimators=150, random_state=42).fit(scaler.transform(X_train), y_train)
    return model, scaler, X_test, y_test

def proj_08_rfr_house():
    st.header("08 • Random Forest Regression — House Price")
    model, scaler, X_test, y_test = _train_rfr_house()
    preds = model.predict(scaler.transform(X_test))
    st.write("R2:", r2_score(y_test, preds))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
    defaults = list(X_test.iloc[0])
    inputs = [st.number_input(f"feat_{i+1}", value=float(defaults[i]), key=f"house_{i}") for i in range(X_test.shape[1])]
    if st.button("Predict (RFR)"):
        val = np.array(inputs).reshape(1,-1)
        p = model.predict(scaler.transform(val))[0]
        st.write("Predicted:", float(p))

# -------------------------
# 09 - Decision Tree Regr Salary
# -------------------------
@st.cache_data(show_spinner=False)
def _train_dtr_salary():
    np.random.seed(42)
    N=400
    exp = np.random.randint(0,20,N)
    edu = np.random.choice([0,1,2],N)
    salary = 20000 + exp*2000 + edu*10000 + np.random.normal(0,5000,N)
    df = pd.DataFrame({"experience":exp,"education":edu,"salary":salary})
    X_train, X_test, y_train, y_test = train_test_split(df[["experience","education"]], df["salary"], test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(max_depth=7, random_state=42).fit(X_train,y_train)
    return model, X_test, y_test

def proj_09_dtr_salary():
    st.header("09 • Decision Tree Regression — Salary")
    model, X_test, y_test = _train_dtr_salary()
    preds = model.predict(X_test)
    st.write("R2:", r2_score(y_test, preds))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
    exp = st.number_input("Years exp", value=5)
    edu = st.selectbox("Education", [0,1,2], format_func=lambda x: {0:"Bachelors",1:"Masters",2:"PhD"}[x])
    if st.button("Predict (DTR)"):
        p = model.predict([[exp,edu]])[0]
        st.write("Predicted Salary:", f"${p:,.0f}")

# -------------------------
# 10 - SVR Car Price
# -------------------------
@st.cache_data(show_spinner=False)
def _train_svr_car():
    np.random.seed(42)
    N=500
    age = np.random.randint(0,15,N)
    km = np.random.randint(0,200000,N)
    hp = np.random.randint(60,300,N)
    price = 30000 - age*1200 - km*0.03 + hp*50 + np.random.normal(0,2000,N)
    df = pd.DataFrame({"age":age,"km":km,"hp":hp,"price":price})
    X_train, X_test, y_train, y_test = train_test_split(df[["age","km","hp"]], df["price"], test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    model = SVR(kernel='rbf', C=100, epsilon=100).fit(scaler.transform(X_train), y_train)
    return model, scaler, X_test, y_test

def proj_10_svr_car():
    st.header("10 • SVR — Car Price")
    model, scaler, X_test, y_test = _train_svr_car()
    preds = model.predict(scaler.transform(X_test))
    st.write("R2:", r2_score(y_test, preds))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
    age = st.number_input("Age", value=5)
    km = st.number_input("Mileage", value=50000)
    hp = st.number_input("Horsepower", value=150)
    if st.button("Predict (SVR)"):
        p = model.predict(scaler.transform([[age,km,hp]]))[0]
        st.write("Predicted Price:", f"${p:,.0f}")

# -------------------------
# 11 - KMeans Segmentation
# -------------------------
@st.cache_data(show_spinner=False)
def _make_kmeans_data():
    np.random.seed(42)
    X = np.vstack([
        np.random.normal([30,30],5,(200,2)),
        np.random.normal([70,60],7,(200,2)),
        np.random.normal([40,80],6,(200,2))
    ])
    return X

def proj_11_kmeans_seg():
    st.header("11 • KMeans — Customer Segmentation")
    X = _make_kmeans_data()
    k = st.slider("k clusters", 2, 6, 3)
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    fig, ax = plt.subplots()
    ax.scatter(X[:,0], X[:,1], c=km.labels_, cmap="tab10", alpha=0.6)
    st.pyplot(fig)
    x = st.number_input("Feature 1", value=35.0)
    y = st.number_input("Feature 2", value=30.0)
    if st.button("Predict Cluster"):
        st.write("Cluster:", int(km.predict([[x,y]])[0]))

# -------------------------
# 12 - Hierarchical GDP
# -------------------------
@st.cache_data(show_spinner=False)
def _make_gdp_data():
    np.random.seed(42)
    data = pd.DataFrame({
        "country":[f"Country_{i}" for i in range(1,31)],
        "gdp_pc": np.random.uniform(5000,60000,30),
        "growth": np.random.uniform(-1,6,30),
        "popdens": np.random.uniform(10,1000,30)
    }).set_index("country")
    return data

def proj_12_hier_gdp():
    st.header("12 • Hierarchical Clustering — Country GDP")
    data = _make_gdp_data()
    st.write(data.head(8))
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    Z = linkage(data.values, method='ward')
    fig, ax = plt.subplots(figsize=(10,4))
    dendrogram(Z, labels=data.index.tolist(), leaf_rotation=90)
    st.pyplot(fig)
    k = st.slider("clusters", 2, 6, 3)
    labels = fcluster(Z, k, criterion='maxclust')
    data['cluster'] = labels
    st.write(data['cluster'].value_counts())

# -------------------------
# 13 - DBSCAN IoT Anomaly
# -------------------------
@st.cache_data(show_spinner=False)
def _make_iot():
    np.random.seed(42)
    normal = np.random.normal(0,1,(500,2))
    anomalies = np.random.normal(8,0.5,(12,2))
    X = np.vstack([normal, anomalies])
    return X

def proj_13_dbscan_iot():
    st.header("13 • DBSCAN — IoT Anomaly Detection")
    X = _make_iot()
    eps = st.slider("eps", 0.1, 3.0, 0.9)
    min_samples = st.slider("min_samples", 1, 10, 5)
    Xs = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(Xs)
    labels = db.labels_
    fig, ax = plt.subplots()
    ax.scatter(X[:,0], X[:,1], c=(labels==-1), cmap='coolwarm', alpha=0.7)
    st.pyplot(fig)
    st.write("Anomalies detected:", int((labels==-1).sum()))

# -------------------------
# 14 - PCA + Logistic Regression
# -------------------------
@st.cache_data(show_spinner=False)
def _train_pca_logreg():
    try:
        d = load_dataset("wine", split="train[:1000]")
        df = pd.DataFrame(d)
        X = df.select_dtypes(include=[np.number]).drop(columns=["target"], errors='ignore').fillna(0)
        y = df["target"] if "target" in df.columns else (np.random.rand(len(X))>0.5).astype(int)
    except Exception:
        from sklearn.datasets import load_wine
        d = load_wine(as_frame=True)
        X = d.data; y = (d.target > d.target.mean()).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)
    pca = PCA(n_components=min(5, X.shape[1])).fit(X_train)
    Xtr = pca.transform(X_train); Xte = pca.transform(X_test)
    clf = LogisticRegression(max_iter=500).fit(Xtr, y_train)
    return pca, clf, X_test, y_test

def proj_14_pca_logreg():
    st.header("14 • PCA + Logistic Regression")
    pca, clf, X_test, y_test = _train_pca_logreg()
    preds = clf.predict(pca.transform(X_test))
    st.write("Accuracy:", accuracy_score(y_test, preds))
    st.text(classification_report(y_test, preds, zero_division=0))

# -------------------------
# 15 - Ensemble Voting
# -------------------------
@st.cache_data(show_spinner=False)
def _train_ensemble():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(as_frame=True)
    X = d.data; y = d.target
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(max_iter=500)
    svc = SVC(probability=True)
    vc = VotingClassifier([("rf",rf),("lr",lr),("svc",svc)], voting="soft")
    vc.fit(scaler.transform(X_train), y_train)
    return vc, scaler, X_test, y_test

def proj_15_ensemble():
    st.header("15 • Ensemble Voting Classifier")
    model, scaler, X_test, y_test = _train_ensemble()
    preds = model.predict(scaler.transform(X_test))
    st.write("Accuracy:", accuracy_score(y_test, preds))
    st.text(classification_report(y_test, preds, zero_division=0))
