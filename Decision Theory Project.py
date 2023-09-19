import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def Eig(A):

    eig_val = np.linalg.eig(A)[0].max()
    eig_vec = np.linalg.eig(A)[1][:,0]
    eig_vec = eig_vec/eig_vec.sum()
    
    return eig_vec, eig_val

def CR(A):
    
    n = A.shape[0]
    _,lambda_max = Eig(A)
    
    CI = (lambda_max-n)/(n-1)
    RCI = [1, 1, .58, .9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.46]
    
    CR = CI/RCI[n-1]
    
    return CI, CR, CR<=.1


def AHP(Attendee):
    
    ### reading attendee ###
    f = df.iloc[Attendee, 5:20]. to_numpy()
    
    ### Creating Two-By-Two Comparison Matrix for Factors ###
    f2b2 = np.zeros((6,6))

    f2b2[0,1:6] = f2b2[0,1:6] + f[0:5]  
    f2b2[1,2:6] = f2b2[1,2:6] + f[5:9]
    f2b2[2,3:6] = f2b2[2,3:6] + f[9:12]
    f2b2[3,4:6] = f2b2[3,4:6] + f[12:14]
    f2b2[4,5:6] = f2b2[4,5:6] + f[14:15]

    for i in range(f2b2.shape[0]):
        for j in range(f2b2.shape[1]):
            if i==j:
                f2b2[i][j] += 1
            elif i>j:
                f2b2[i][j] += 1/f2b2[j][i]
            else:
                pass

    ### Creating Two-By-Two Comparison Matrix for Alternatives Per Factor ###
    lf=[]

    for k in range(6):

        f = df.iloc[Attendee, 20+(6*k):26+(6*k)]. to_numpy()

        k2b2 = np.zeros((4,4))

        k2b2[0,1:4] = k2b2[0,1:4] + f[0:3]  
        k2b2[1,2:4] = k2b2[1,2:4] + f[3:5]
        k2b2[2,3:4] = k2b2[2,3:4] + f[5:6]

        for i in range(k2b2.shape[0]):
            for j in range(k2b2.shape[1]):
                if i==j:
                    k2b2[i][j] += 1
                elif i>j:
                    k2b2[i][j] += 1/k2b2[j][i]
                else:
                    pass

        lf.append(k2b2)

    es2b2 = lf[0]
    fs2b2 = lf[1]
    in2b2 = lf[2]
    ev2b2 = lf[3]
    sj2b2 = lf[4]
    lg2b2 = lf[5]

    ### Starting Calculations ###
    CI_0,CR_0,_ = CR(f2b2)
    CI_1,CR_1,_ = CR(es2b2)
    CI_2,CR_2,_ = CR(fs2b2)
    CI_3,CR_3,_ = CR(in2b2)
    CI_4,CR_4,_ = CR(ev2b2)
    CI_5,CR_5,_ = CR(sj2b2)
    CI_6,CR_6,_ = CR(lg2b2)
    
    ### Calculating OCR ###
    W_f2b2,_ = Eig(f2b2)

    CI_vec = np.array([CI_0, CI_1, CI_2, CI_3, CI_4, CI_5, CI_6])
    RCI_vec = np.array([1.24, .9, .9, .9, .9, .9, .9])
    W_vec = np.array([1, W_f2b2[0], W_f2b2[1], W_f2b2[2], W_f2b2[3], W_f2b2[4], W_f2b2[5]])
    
    OCR = (np.dot(W_vec,CI_vec))/(np.dot(W_vec,RCI_vec))
    
    ### Calculating P ###
    W_es2b2,_ = Eig(es2b2)
    W_fs2b2,_ = Eig(fs2b2)
    W_in2b2,_ = Eig(in2b2)
    W_ev2b2,_ = Eig(ev2b2)
    W_sj2b2,_ = Eig(sj2b2)
    W_lg2b2,_ = Eig(lg2b2)
    
    S = np.concatenate(([W_es2b2], [W_fs2b2], [W_in2b2], [W_ev2b2], [W_sj2b2], [W_lg2b2]))
    P = np.matmul([W_f2b2], S)
    
    F = np.argsort(W_f2b2)[::-1][:6]
    A = np.argsort(P[0])[::-1][:4]
    
    return CR_0, CR_1, CR_2, CR_3, CR_4, CR_5, CR_6, OCR, P, A, F

### Loading the form data ###
df = pd.read_csv(r"G:\University\Terme 7\Assignments\DM\Project\Data\Data_Coded.csv")

### Creating our dataframe ###
df_result = pd.DataFrame(columns=['Attendee', "Gender", "Age", "Education","Job",
                                  "Factor1", "Factor2", "Factor3", "Factor4", "Factor5",  "Factor6",
                                  "Alternative1", "Alternative2", "Alternative3", "Alternative4",
                                  'CR_0', 'CR_1', 'CR_2', 'CR_3', 'CR_4', 'CR_5', 'CR_6', 'OCR'])


### Calculating the desired matrices for each attendee ###
for l in range(len(df)):
    
    R = AHP(Attendee=l)
    
    New_Attendee = {'Attendee':l+1,
                    "Gender":df["Gender"][l],
                    "Age":df["Age"][l],
                    "Education":df["Education"][l],
                    "Job":df["Job"][l],
                    "Factor1": np.real(R[10][0]),
                    "Factor2": np.real(R[10][1]),
                    "Factor3": np.real(R[10][2]),
                    "Factor4": np.real(R[10][3]),
                    "Factor5": np.real(R[10][4]),
                    "Factor6": np.real(R[10][5]),
                    "Alternative1": np.real(R[9][0]),
                    "Alternative2": np.real(R[9][1]),
                    "Alternative3": np.real(R[9][2]),
                    "Alternative4": np.real(R[9][3]),
                    'CR_0': np.real(R[0]),
                    'CR_1': np.real(R[1]),
                    'CR_2': np.real(R[2]),
                    'CR_3': np.real(R[3]),
                    'CR_4': np.real(R[4]),
                    'CR_5': np.real(R[5]),
                    'CR_6': np.real(R[6]),
                    'OCR': np.real(R[7])}
    
    df_result = df_result.append(New_Attendee, ignore_index=True)


### Defining dics to convert coded data ###
factors = {0:"Econoomic Stability",
           1:"Freedom of Speech",
           2:"Insfrastructure",
           3:"Environment",
           4:"Social Justice",
           5:"Legitimate Government"}

Alternatives = {0:"Reform",
                1:"Revolution",
                2:"Coup",
                3:"Do Nothing"}

Gender = {0:"Man",
          1:"Woman"}

Age = {0:"15-20",
       1:"20-25",
       2:"25-30",
       3:"30-40",
       4:"40-50",
       5:"50-60"}

Education = {0:"High School Dropout or Less",
             1:"High School Graduate",
             2:"Bachelor's Degree",
             3:"Master's Degree",
             4:"Doctorate Degree"}

Job = {0:"Student",
       1:"Teacher or Professor",
       2:"Bank or Insurance Employee",
       3:"Engineer",
       4:"Self Employed"}

df_result = df_result.replace({"Gender": Gender,
                               "Age": Age,
                               "Education": Education,
                               "Job": Job,
                               "Factor1": factors,
                               "Factor2": factors,
                               "Factor3": factors,
                               "Factor4": factors,
                               "Factor5": factors,
                               "Factor6": factors,
                               "Alternative1": Alternatives,
                               "Alternative2": Alternatives,
                               "Alternative3": Alternatives,
                               "Alternative4": Alternatives})


### Analyizing consistency ###
df_1 = df_result.copy()

for i in range(0,7):
    for j in range(len(df_result["CR_" + str(i)])):
        if type(df_result["CR_" + str(i)][j]) == float:
            if df_result["CR_" + str(i)][j] <= 0.1:
                df_1["CR_" + str(i)][j] = "Consistent"
            elif df_result["CR_" + str(i)][j] > 0.1:
                df_1["CR_" + str(i)][j] = "Inconsistent"

   
for z in range(len(df_result["OCR"])):
    if type(df_result["OCR"][z]) == float:
        if df_result["OCR"][z] <= 0.1:
            df_1["OCR"][z] = "Consistent"
        elif df_result["OCR"][z] > 0.1:
            df_1["OCR"][z] = "Inconsistent"

df_C = df_1[["CR_0", "CR_1", "CR_2", "CR_3", "CR_4", "CR_5", "CR_6", "OCR"]]

C = {"Consistent":1,
     "Inconsistent":0}

df_C = df_C.replace({"CR_0": C,
                     "CR_1": C,
                     "CR_2": C,
                     "CR_3": C,
                     "CR_4": C,
                     "CR_5": C,
                     "CR_6": C,
                     "OCR": C})

Con = []

for i in range(len(df_C["CR_0"])):
    a = (df_C.iloc[i].sum()/8)*100
    Con.append(a)

df_1["Consistency %"] = Con


### Drawing plots showing consistency rates amongst different groups ###
df_1_G = df_1.groupby(["Gender"]).mean()["Consistency %"].reset_index(drop=False)
sns.barplot(x="Gender", y="Consistency %", data=df_1_G, order=list(Gender.values()))
plt.xticks(rotation=0)
plt.title("Inconsistencies Among Different Genders",fontdict={'fontweight':'bold','fontsize':15},pad=10)
plt.xlabel('Gender',fontdict={'fontweight':'bold','fontsize':10},labelpad=5)
plt.ylabel('Average Consistency %',fontdict={'fontweight':'bold','fontsize':10},labelpad=5)
plt.savefig(r'G:\University\Terme 7\Assignments\DM\Project\Code Exports\Inconsistencies Among Different Genders.png',bbox_inches = 'tight',dpi=600)
plt.close()


df_1_A = df_1.groupby(["Age"]).mean()["Consistency %"].reset_index(drop=False)
sns.barplot(x="Age", y="Consistency %", data=df_1_A, order=list(Age.values()))
plt.xticks(rotation=0)
plt.title("Inconsistencies Among Different Age Groups",fontdict={'fontweight':'bold','fontsize':15},pad=10)
plt.xlabel('Age',fontdict={'fontweight':'bold','fontsize':10},labelpad=5)
plt.ylabel('Average Consistency %',fontdict={'fontweight':'bold','fontsize':10},labelpad=5)
plt.savefig(r'G:\University\Terme 7\Assignments\DM\Project\Code Exports\Inconsistencies Among Different Age Groups.png',bbox_inches = 'tight',dpi=600)
plt.close()


df_1_E = df_1.groupby(["Education"]).mean()["Consistency %"].reset_index(drop=False)
sns.barplot(x="Education", y="Consistency %", data=df_1_E, order=list(Education.values()))
plt.xticks(rotation=90)
plt.title("Inconsistencies Among Different Education Levels",fontdict={'fontweight':'bold','fontsize':15},pad=10)
plt.xlabel('Education',fontdict={'fontweight':'bold','fontsize':10},labelpad=-10)
plt.ylabel('Average Consistency %',fontdict={'fontweight':'bold','fontsize':10},labelpad=5)
plt.savefig(r'G:\University\Terme 7\Assignments\DM\Project\Code Exports\Inconsistencies Among Different Education Levels.png',bbox_inches = 'tight',dpi=600)
plt.close()


df_1_J = df_1.groupby(["Job"]).mean()["Consistency %"].reset_index(drop=False)
sns.barplot(x="Job", y="Consistency %", data=df_1_J, order=list(Job.values()))
plt.xticks(rotation=90)
plt.title("Inconsistencies Among Different Jobs",fontdict={'fontweight':'bold','fontsize':15},pad=10)
plt.xlabel('Job',fontdict={'fontweight':'bold','fontsize':10},labelpad=5)
plt.ylabel('Average Consistency %',fontdict={'fontweight':'bold','fontsize':10},labelpad=5)
plt.savefig(r'G:\University\Terme 7\Assignments\DM\Project\Code Exports\Inconsistencies Among Different Jobs.png',bbox_inches = 'tight',dpi=600)
plt.close()

### Saving our initial dataframe with n=0.1 for consistency degree of freedom ###
df_1.to_csv(r'G:\University\Terme 7\Assignments\DM\Project\Code Exports\DM Project Concluded Data (0.1).csv',index=False)

### Analyzing different amounts of n and its impact on number of consistencies ###
df_clone1 = df_result.copy()

rn = np.arange(0.1,0.205,0.005)

Con_rate = []

for h in rn:
    df_clone = df_clone1.copy()
    for i in range(0,7):
        for j in range(len(df_result["CR_" + str(i)])):
            if type(df_result["CR_" + str(i)][j]) == float:
                if df_result["CR_" + str(i)][j] <= h:
                    df_clone["CR_" + str(i)][j] = "Consistent"
                elif df_result["CR_" + str(i)][j] > h:
                    df_clone["CR_" + str(i)][j] = "Inconsistent"
    
       
    for z in range(len(df_result["OCR"])):
        if type(df_result["OCR"][z]) == float:
            if df_result["OCR"][z] <= h:
                df_clone["OCR"][z] = "Consistent"
            elif df_result["OCR"][z] > h:
                df_clone["OCR"][z] = "Inconsistent"
    
    df_C = df_clone[["CR_0", "CR_1", "CR_2", "CR_3", "CR_4", "CR_5", "CR_6", "OCR"]]
    
    C = {"Consistent":1,
         "Inconsistent":0}
    
    df_C = df_C.replace({"CR_0": C,
                         "CR_1": C,
                         "CR_2": C,
                         "CR_3": C,
                         "CR_4": C,
                         "CR_5": C,
                         "CR_6": C,
                         "OCR": C})
    
    Con = []
    
    for i in range(len(df_C["CR_0"])):
        a = df_C.iloc[i].sum()/8
        Con.append(a)
    
    df_clone["Consistency %"] = Con
    
    f = float(df_clone.loc[df_clone["OCR"] == "Consistent"].count()["Attendee"])
    Con_rate.append(f)
    
   
Con_rate    


### Drawing a line chart indicating the impact of n on number of consistencies ###
plt.plot(rn,Con_rate,marker = 'o',markersize=3,mec='r',mfc='r')
plt.xticks(rn,size=10, rotation=45)
plt.yticks(np.arange(min(Con_rate),max(Con_rate)+1,2),size=10)
plt.tight_layout()
plt.grid()

plt.title("Count of Counsistent Responses",fontdict={'fontweight':'bold','fontsize':15},pad=10)
plt.xlabel('Degree of Freedom',fontdict={'fontweight':'bold','fontsize':12},labelpad=5)
plt.ylabel('Count',fontdict={'fontweight':'bold','fontsize':12},labelpad=5)

plt.savefig(r'G:\University\Terme 7\Assignments\DM\Project\Code Exports\Count of Counsistent Responses.png',bbox_inches = 'tight',dpi=600)
plt.close()

### Creating a new dataframe using the median of n in previous section ###
df_2 = df_result.copy()
ch = rn[int(round(((len(Con_rate)-1)/2),0))]

for i in range(0,7):
    for j in range(len(df_result["CR_" + str(i)])):
        if type(df_result["CR_" + str(i)][j]) == float:
            if df_result["CR_" + str(i)][j] <= ch:
                df_2["CR_" + str(i)][j] = "Consistent"
            elif df_result["CR_" + str(i)][j] > ch:
                df_2["CR_" + str(i)][j] = "Inconsistent"

   
for z in range(len(df_result["OCR"])):
    if type(df_result["OCR"][z]) == float:
        if df_result["OCR"][z] <= ch:
            df_2["OCR"][z] = "Consistent"
        elif df_result["OCR"][z] > ch:
            df_2["OCR"][z] = "Inconsistent"

df_C = df_2[["CR_0", "CR_1", "CR_2", "CR_3", "CR_4", "CR_5", "CR_6", "OCR"]]

C = {"Consistent":1,
     "Inconsistent":0}

df_C = df_C.replace({"CR_0": C,
                     "CR_1": C,
                     "CR_2": C,
                     "CR_3": C,
                     "CR_4": C,
                     "CR_5": C,
                     "CR_6": C,
                     "OCR": C})

Con = []

for i in range(len(df_C["CR_0"])):
    a = (df_C.iloc[i].sum()/8)*100
    Con.append(a)

df_2["Consistency %"] = Con


### Drawing plots showing consistency rates amongst different groups after correction###
df_2_G = df_2.groupby(["Gender"]).mean()["Consistency %"].reset_index(drop=False)
sns.barplot(x="Gender", y="Consistency %", data=df_2_G, order=list(Gender.values()))
plt.xticks(rotation=0)
plt.title("Inconsistencies Among Different Genders After Correction",fontdict={'fontweight':'bold','fontsize':15},pad=10)
plt.xlabel('Gender',fontdict={'fontweight':'bold','fontsize':10},labelpad=5)
plt.ylabel('Average Consistency %',fontdict={'fontweight':'bold','fontsize':10},labelpad=5)
plt.savefig(r'G:\University\Terme 7\Assignments\DM\Project\Code Exports\Inconsistencies Among Different Genders After Correction.png',bbox_inches = 'tight',dpi=600)
plt.close()


df_2_A = df_2.groupby(["Age"]).mean()["Consistency %"].reset_index(drop=False)
sns.barplot(x="Age", y="Consistency %", data=df_2_A, order=list(Age.values()))
plt.xticks(rotation=0)
plt.title("Inconsistencies Among Different Age Groups After Correction",fontdict={'fontweight':'bold','fontsize':15},pad=10)
plt.xlabel('Age',fontdict={'fontweight':'bold','fontsize':10},labelpad=5)
plt.ylabel('Average Consistency %',fontdict={'fontweight':'bold','fontsize':10},labelpad=5)
plt.savefig(r'G:\University\Terme 7\Assignments\DM\Project\Code Exports\Inconsistencies Among Different Age Groups After Correction.png',bbox_inches = 'tight',dpi=600)
plt.close()


df_2_E = df_2.groupby(["Education"]).mean()["Consistency %"].reset_index(drop=False)
sns.barplot(x="Education", y="Consistency %", data=df_2_E, order=list(Education.values()))
plt.xticks(rotation=90)
plt.title("Inconsistencies Among Different Education Levels After Correction",fontdict={'fontweight':'bold','fontsize':15},pad=10)
plt.xlabel('Education',fontdict={'fontweight':'bold','fontsize':10},labelpad=-10)
plt.ylabel('Average Consistency %',fontdict={'fontweight':'bold','fontsize':10},labelpad=5)
plt.savefig(r'G:\University\Terme 7\Assignments\DM\Project\Code Exports\Inconsistencies Among Different Education Levels After Correction.png',bbox_inches = 'tight',dpi=600)
plt.close()


df_2_J = df_2.groupby(["Job"]).mean()["Consistency %"].reset_index(drop=False)
sns.barplot(x="Job", y="Consistency %", data=df_2_J, order=list(Job.values()))
plt.xticks(rotation=90)
plt.title("Inconsistencies Among Different Jobs After Correction",fontdict={'fontweight':'bold','fontsize':15},pad=10)
plt.xlabel('Job',fontdict={'fontweight':'bold','fontsize':10},labelpad=5)
plt.ylabel('Average Consistency %',fontdict={'fontweight':'bold','fontsize':10},labelpad=5)
plt.savefig(r'G:\University\Terme 7\Assignments\DM\Project\Code Exports\Inconsistencies Among Different Jobs After Correction.png',bbox_inches = 'tight',dpi=600)
plt.close()

### Saving our second dataframe with n=0.15 for consistency degree of freedom ###
df_2.to_csv(r'G:\University\Terme 7\Assignments\DM\Project\Code Exports\DM Project Concluded Data (0.15).csv',index=False)

### Saving our final dataframe with n=0.15 for consistency degree of freedom with attendees with Consistend answers ###
df_final = df_2.loc[df_2["OCR"] == "Consistent"].reset_index(drop=True)
df_final.to_csv(r'G:\University\Terme 7\Assignments\DM\Project\Code Exports\DM Project Concluded Data (final).csv',index=False)

### Analyzing factors ###
Fac = list(factors.values())

Fac_df = pd.DataFrame(columns=["Factor1", "Factor2", "Factor3", "Factor4", "Factor5", "Factor6"])

for i in range(len(Fac)):
    PF = []
    for j in range(1,7): 
        pf = round(sum(df_final["Factor" + str(j)] == Fac[i])/len(df_final["Factor" + str(j)]),3)*100
        PF.append(pf)
        
    Fac_df.loc[Fac[i]] = PF
Fac_df.to_csv(r'G:\University\Terme 7\Assignments\DM\Project\Code Exports\Factors Analysis (All).csv',index=True)    

### Analyzing alternatives ###
Alt = list(Alternatives.values())

Alt_df = pd.DataFrame(columns=["Alternative1", "Alternative2", "Alternative3", "Alternative4"])

for i in range(len(Alt)):
    PA = []
    for j in range(1,5): 
        pa = round(sum(df_final["Alternative" + str(j)] == Alt[i])/len(df_final["Factor" + str(j)]),3)*100
        PA.append(pa)
        
    Alt_df.loc[Alt[i]] = PA
Alt_df.to_csv(r'G:\University\Terme 7\Assignments\DM\Project\Code Exports\Alternatives Analysis (All).csv',index=True)    

### Analyzing factors among men ###
df_final_man = df_final.loc[df_final["Gender"] == "Man"]

Fac_df_M = pd.DataFrame(columns=["Factor1", "Factor2", "Factor3", "Factor4", "Factor5", "Factor6"])

for i in range(len(Fac)):
    PF_M = []
    for j in range(1,7): 
        pf_m = round(sum(df_final_man["Factor" + str(j)] == Fac[i])/len(df_final_man["Factor" + str(j)]),3)*100
        PF_M.append(pf_m)
        
    Fac_df_M.loc[Fac[i]] = PF_M
Fac_df_M.to_csv(r'G:\University\Terme 7\Assignments\DM\Project\Code Exports\Factors Analysis (Men).csv',index=True) 

### Analyzing alternatives among men ###
Alt_df_M = pd.DataFrame(columns=["Alternative1", "Alternative2", "Alternative3", "Alternative4"])

for i in range(len(Alt)):
    PA_M = []
    for j in range(1,5): 
        pa_m = round(sum(df_final_man["Alternative" + str(j)] == Alt[i])/len(df_final_man["Factor" + str(j)]),3)*100
        PA_M.append(pa_m)
        
    Alt_df_M.loc[Alt[i]] = PA_M
Alt_df_M.to_csv(r'G:\University\Terme 7\Assignments\DM\Project\Code Exports\Alternatives Analysis (Men).csv',index=True) 

### Analyzing factors amon women ###
df_final_woman = df_final.loc[df_final["Gender"] == "Woman"]

Fac_df_W = pd.DataFrame(columns=["Factor1", "Factor2", "Factor3", "Factor4", "Factor5", "Factor6"])

for i in range(len(Fac)):
    PF_W = []
    for j in range(1,7): 
        pf_w = round(sum(df_final_woman["Factor" + str(j)] == Fac[i])/len(df_final_woman["Factor" + str(j)]),3)*100
        PF_W.append(pf_w)
        
    Fac_df_W.loc[Fac[i]] = PF_W
Fac_df_W.to_csv(r'G:\University\Terme 7\Assignments\DM\Project\Code Exports\Factors Analysis (Women).csv',index=True)

### Analyzing alternatives among women ###
Alt_df_W = pd.DataFrame(columns=["Alternative1", "Alternative2", "Alternative3", "Alternative4"])

for i in range(len(Alt)):
    PA_W = []
    for j in range(1,5): 
        pa_w = round(sum(df_final_woman["Alternative" + str(j)] == Alt[i])/len(df_final_woman["Factor" + str(j)]),3)*100
        PA_W.append(pa_w)
        
    Alt_df_W.loc[Alt[i]] = PA_W
Alt_df_W.to_csv(r'G:\University\Terme 7\Assignments\DM\Project\Code Exports\Alternatives Analysis (Women).csv',index=True)





