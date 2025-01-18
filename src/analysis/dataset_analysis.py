import streamlit as st
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns
from scipy.stats import skew
from scipy.stats import kurtosis
from sklearn.metrics import mutual_info_score


# function to apply descaling to an array
def descaling (arry):
    miny = np.min(arry, axis=0)
    maxy = np.max(arry, axis=0)
    desarry = np.zeros(len(arry))
    for i in range (0,len(arry)):
        desarry [i] = (arry[i] -  miny) / (maxy - miny)

    return desarry

# function to apply standardization to an array
def standardization (arry):
    meany = np.mean(arry, axis=0)
    stdy = np.std(arry, axis=0)
    stdarry = np.zeros(len(arry))
    for i in range (0,len(arry)):
        stdarry [i] = (arry[i] -  meany) / stdy

    return stdarry

def print_information_and_statistics():
    #load raw dataset
    data = './src/data/Data_Science_Salaries.csv'
    df = pd.read_csv(data)
    df1 = df
    df1=df1.drop(index=0, axis=1)

    #drop duplicates, keeping only one of them

    df1=df1.drop_duplicates(keep='first')
    dfr=df1
    
    #separate the target feature y from the rest X and analyse y
    st.subheader("Statistics")
    y=dfr['salary_in_usd']
    col1,col2=st.columns(2)
    with col1.container(border=True):
        minimum_salary=np.min(y)
        maximum_salary=np.max(y)
        average_salary=np.average(y)
        mean_salary=y.mean()
        median_salary=y.median()
        st.write(f"Total Records: {len(y)}")
        st.write(f"Minimum: ${minimum_salary:,.2f}")
        st.write(f"Maximum: ${maximum_salary:,.2f}")
        st.write(f"Average: ${average_salary:,.2f}")
        st.write(f"Mean: ${mean_salary:,.2f}")
        st.write(f"Median: ${median_salary:,.2f}")
        st.write(f"Standard Deviation: ${np.std(y):,.2f}")
    with col2.container(border=True):
        # add the salary category feature
        iqr25=np.percentile(y, 25)
        lmsedge=np.percentile(y, 33)
        mhsedge=np.percentile(y, 66)
        iqr75=np.percentile(y, 75)
        st.write(f"Skewness: {skew(y, axis=0, bias=True):.6f}")
        st.write(f"Kurtosis: {kurtosis(y, axis=0, bias=True):.6f}")
        st.write("")
        st.write("")
        st.write(f"25th percentile of arr : ${iqr25:,.2f}")
        st.write(f"33rd percentile of arr : ${lmsedge:,.2f}")
        st.write(f"66th percentile of arr : ${mhsedge:,.2f}")
        st.write(f"75th percentile of arr : ${iqr75:,.2f}")
    
    # Plot the salary distribution
    fig,ax=plt.subplots(figsize=(10, 6))
    sns.histplot(y, kde=True, bins=30, color='blue', edgecolor='black')
    ax.axvline(mean_salary, color='red', linestyle='dashed', linewidth=1, label=f'Mean: ${mean_salary:,.2f}')
    ax.axvline(median_salary, color='green', linestyle='dashed', linewidth=1, label=f'Median: ${median_salary:,.2f}')
    ax.set_title('Salary Distribution (USD)', fontsize=16)
    ax.set_xlabel('Salary (USD)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.close(fig)
    st.pyplot(fig)

    st.write("Dataset Entries Scatter Plot")
    #plot scatter original data
    df1["id"] = df1.index
    fig,ax=plt.subplots(figsize=(15,10))
    ax.scatter(df1['id'], df1['salary_in_usd'])
    plt.close(fig)
    st.pyplot(fig)

    # create a list of our conditions
    def salary_group(value):
        if value < lmsedge:
            return "L"
        if lmsedge <= value < mhsedge:
            return "M"
        elif value >= mhsedge:
            return "H"

    dfr['salary_group'] = dfr['salary_in_usd'].map(salary_group)

    st.divider()
    
    # generate 3 csv files, one for regression with the salary_in_usd values, one for classification
    #with the salary_group and one for both with categorical features transformed to numerical ones 
    dfc = dfr
    dfc = dfc.drop(['salary_in_usd'], axis=1)
    classificationfile = './src/data/DS_salaries_classification.csv'
    regressionfile = './src/data/DS_salaries_regression.csv'
    filewithnumerical = './src/data/DS_salaries_regression_numerical.csv'

    if not os.path.isfile(regressionfile):
        dfr.to_csv(regressionfile)
    if not os.path.isfile(classificationfile):
        dfc.to_csv(classificationfile)     

    st.write("Mapped Dataset Previews:")
    
    st.write(dfr.head())

    # transform categorical features into numerical ones using target encoding applying also scaling
    #year
    yearv = dfr.groupby('work_year', as_index=False)['salary_in_usd'].mean()
    arry = yearv["salary_in_usd"].to_numpy()
    desarray = descaling (arry)
    yearv['des_year']=desarray
    st.write(yearv)
    # Create the mapping dictionary from DF1
    yearmap = yearv.set_index('work_year')['des_year'].to_dict()
    # Add the "des_year" column to DF2 based on the mapping
    dfr['nd_work_year'] = dfr['work_year'].map(yearmap)


    #experience_level
    elv = dfr.groupby('experience_level', as_index=False)['salary_in_usd'].mean()
    arrel = elv["salary_in_usd"].to_numpy()
    desarrel = descaling (arrel)
    elv['des_el']=desarrel
    st.write (elv)
    elmap = elv.set_index('experience_level')['des_el'].to_dict()
    dfr['nd_experience_level'] = dfr['experience_level'].map(elmap)


    #employment_type
    etv = dfr.groupby('employment_type', as_index=False)['salary_in_usd'].mean()
    arret = etv["salary_in_usd"].to_numpy()
    desarret = descaling (arret)
    etv['des_et']=desarret
    st.write (etv)
    etmap = etv.set_index('employment_type')['des_et'].to_dict()
    dfr['nd_employment_type'] = dfr['employment_type'].map(etmap)

    #job_title
    jtv = dfr.groupby('job_title', as_index=False)['salary_in_usd'].mean()
    arrjt = jtv["salary_in_usd"].to_numpy()
    desarrjt = descaling (arrjt)
    jtv['des_jt']=desarrjt
    st.write (jtv)
    jtmap = jtv.set_index('job_title')['des_jt'].to_dict()
    dfr['nd_job_title'] = dfr['job_title'].map(jtmap)

    #employee_residence
    erv = dfr.groupby('employee_residence', as_index=False)['salary_in_usd'].mean()
    arrer = erv["salary_in_usd"].to_numpy()
    desarrer = descaling (arrer)
    erv['des_er']=desarrer
    st.write (erv)
    ermap = erv.set_index('employee_residence')['des_er'].to_dict()
    dfr['nd_employee_residence'] = dfr['employee_residence'].map(ermap)

    #remote_ratio
    rrv = dfr.groupby('remote_ratio', as_index=False)['salary_in_usd'].mean()
    arrrr = rrv["salary_in_usd"].to_numpy()
    desarrrr = descaling (arrrr)
    rrv['des_rr']=desarrrr
    st.write (rrv)
    rrmap = rrv.set_index('remote_ratio')['des_rr'].to_dict()
    dfr['nd_remote_ratio'] = dfr['remote_ratio'].map(rrmap)

    #company_location
    clv = dfr.groupby('company_location', as_index=False)['salary_in_usd'].mean()
    arrcl = clv["salary_in_usd"].to_numpy()
    desarrcl = descaling (arrcl)
    clv['des_cl']=desarrcl
    st.write (clv)
    clmap = clv.set_index('company_location')['des_cl'].to_dict()
    dfr['nd_company_location'] = dfr['company_location'].map(clmap)


    #company_size
    csv = dfr.groupby('company_size', as_index=False)['salary_in_usd'].mean()
    arrcs = csv["salary_in_usd"].to_numpy()
    desarrcs = descaling (arrcs)
    csv['des_cs']=desarrcs
    st.write (csv)
    csmap = csv.set_index('company_size')['des_cs'].to_dict()
    dfr['nd_company_size'] = dfr['company_size'].map(csmap)

    st.write (dfr.head())
    if not os.path.isfile(filewithnumerical):
        # save the cleaned dataset with also the descaled numerical feature
        dfr.to_csv(filewithnumerical)



    #remove the salary_in_usd and salary_group features
    x=dfr.drop(['salary_in_usd'], axis=1)
    x=x.drop(['salary_group'], axis=1)
    x=x[[col for col in x.columns if col.startswith("nd_")]]

    co=x.columns
    nco=len(co)
    #st.write(co)


    st.divider()

    st.write("Mutual Information Scores:")
    col1,col2=st.columns([1,5])
    #Compute the mutual information for every pair of features
    matrixmi= np.zeros([nco, nco])
    i=0

    for c1 in co:
        j=0
        for c2 in co:
            matrixmi[i][j]= round(mutual_info_score(x[c1], x[c2]),2)
            #st.write(c1, c2, i, j, matrixmi[i][j])
            j=j+1
        i=i+1
    with col2:
        fig,ax=plt.subplots(figsize=(15,8))
        sns.heatmap(matrixmi,annot=True,cmap="coolwarm",ax=ax)
        ax.imshow(matrixmi)
        ax.set_title("Heat-map of the Mutual Information of the features of the Data scientist job roles dataset")
        plt.close(fig)
        st.pyplot(fig)
        #st.write(matrixmi)

    with col1:
        #Compute the mutual information for every feature and the target
        vectmift= np.zeros([nco])
        i=0

        for c1 in co:
            vectmift[i] = round(mutual_info_score(x[c1], y),2)
            i=i+1
        
        st.write(vectmift)
    
    st.divider()




def perform():
    st.title("Dataset Information and Statistics")
    st.write("Some information and statistics about the used dataset:")
    
    with st.spinner("performing data analysis, please wait..."):
        with st.container(border=True):
            print_information_and_statistics()

