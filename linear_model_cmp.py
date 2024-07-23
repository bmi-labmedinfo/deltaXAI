from deltaxai import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import shap
import pandas as pd
############################# functions #############################################################################
def plot_rank(vals,featNames,figname):
    plt.rcParams.update({'font.size': 11})
    fig, ax = plt.subplots()
    idx = vals.argsort()[::-1]
    vals = vals[idx]
    featNames = featNames[idx]
    bars = ax.barh(y=np.arange(len(vals)),
                   width=vals)
    ax.set_yticks(np.arange(len(vals)), featNames)
    ax.invert_yaxis()
    ax.set_xlabel("mean(SHAP value)")
    plt.savefig(figname + ".png", bbox_inches='tight', dpi=250)
    plt.clf()


seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
N = 10000 #sample size
############################### model 1 ##############################################################################

means = np.zeros(3)
sigma = np.eye(3)
names = ["X1","X2","X3"]
#dumb fit
Xtr = np.random.multivariate_normal(means,sigma,size=N)
Ytr = np.dot(Xtr,np.array([100,50,50]))
reg1 = LinearRegression(fit_intercept=False).fit(Xtr, Ytr)
#dataframe of training set
Xtr_df = pd.DataFrame(Xtr,columns=names)
#example 1: both numerical and df
X1 = np.array([0,0,2]).reshape((1,3))
X1_df = pd.DataFrame(X1,columns=names)
#shapley
explainer1 = shap.Explainer(reg1.predict,Xtr)
shap_values = explainer1(X1)
plot_rank(np.squeeze(shap_values.values),Xtr_df.columns.to_numpy(),"LinModOuts/m1e1_shap" )
#delta method
delta_expl1 = DeltaExplainer(reg1,Xtr_df,Nboot=500,saveDistr=True,useParallel=False)#
res1 = delta_expl1.explain(X1_df,np.array([100]))
res1.plotRanking("LinModOuts/m1e1_delta")
res1.plotEffects("LinModOuts/m1e1_deltaE")

######### example 2
X2 = np.array([0.1,0.1,0.1]).reshape((1,3))
X2_df = pd.DataFrame(X2,columns=names)
shap_values2 = explainer1(X2)
plot_rank(np.squeeze(shap_values2.values),Xtr_df.columns.to_numpy(),"LinModOuts/m1e2_shap" )

res2 = delta_expl1.explain(X2_df,np.array([20]))
res2.plotRanking("LinModOuts/m1e2_delta")
res2.plotEffects("LinModOuts/m1e2_deltaE")

######### example 3 ####################
X3 = np.array([1,2,2]).reshape((1,3))
X3_df = pd.DataFrame(X3,columns=names)
shap_values3 = explainer1(X3)
plot_rank(np.squeeze(shap_values3.values),Xtr_df.columns.to_numpy(),"LinModOuts/m1e3_shap" )
#
res3 = delta_expl1.explain(X3_df,np.array([300]))
res3.plotRanking("LinModOuts/m1e3_delta")
res3.plotEffects("LinModOuts/m1e3_deltaE")

######### example 3 bis ####################
X3 = np.array([0.4,0.8,0.8]).reshape((1,3))
X3_df = pd.DataFrame(X3,columns=names)
shap_values3 = explainer1(X3)
plot_rank(np.squeeze(shap_values3.values),Xtr_df.columns.to_numpy(),"LinModOuts/m1e3c_shap" )
#
res3 = delta_expl1.explain(X3_df,np.array([120]))
res3.plotRanking("LinModOuts/m1e3c_delta")
res3.plotEffects("LinModOuts/m1e3c_deltaE")


##################################################à model 2 ###########################################################

Ytr2 = np.dot(Xtr,np.array([1000,50,50]))
reg2 = LinearRegression(fit_intercept=False).fit(Xtr, Ytr2)
X4 = np.array([0,0,2]).reshape((1,3))
X4_df = pd.DataFrame(X4,columns=names)
explainer2 = shap.Explainer(reg2.predict,Xtr)

shap_values4 = explainer2(X4)
plot_rank(np.squeeze(shap_values4.values),Xtr_df.columns.to_numpy(),"LinModOuts/m2e1_shap" )
delta_expl2 = DeltaExplainer(reg2,Xtr_df,Nboot=500,saveDistr=True,useParallel=False)

res4 = delta_expl2.explain(X4_df,np.array([100]))
res4.plotRanking("LinModOuts/m2e1_delta")
res4.plotEffects("LinModOuts/m2e1_deltaE")
#
Ytr2b = np.dot(Xtr,np.array([100000,50,50]))
reg2b =LinearRegression(fit_intercept=False).fit(Xtr, Ytr2b)
explainer2b = shap.Explainer(reg2b.predict,Xtr)
shap_values4b = explainer2b(X4)
plot_rank(np.squeeze(shap_values4b.values),Xtr_df.columns.to_numpy(),"LinModOuts/m2e1b_shap" )



############################## model 3 ############################################################################

means = np.zeros(4)
sigma = np.eye(4)
sigma[0,3] = 0.99
sigma[3,0] = 0.99

# #dumb fit
Xtr3 = np.random.multivariate_normal(means,sigma,size=N)
Ytr3 = np.dot(Xtr3,np.array([100,50,50,0]))
reg3 = LinearRegression(fit_intercept=False).fit(Xtr3, Ytr3)
explainer3 = shap.Explainer(reg3.predict,Xtr3)

names3 = ["X1","X2","X3","X4"]
Xtr3_df = pd.DataFrame(Xtr3,columns=names3)

X5 = np.array([0.1,0.1,0.1,1]).reshape((1,4))
X5_df = pd.DataFrame(X5,columns=names3)

shap_values5 = explainer3(X5)
plot_rank(np.squeeze(shap_values5.values),Xtr3_df.columns.to_numpy(),"LinModOuts/m3e1_shap" )
delta_expl3 = DeltaExplainer(reg3,Xtr3_df,Nboot=500,saveDistr=True,useParallel=False)
res5 = delta_expl3.explain(X5_df,np.array([20]))
res5.plotRanking("LinModOuts/m3e1_delta")
res5.plotEffects("LinModOuts/m3e1_deltaE")

means = np.zeros(3)
sigma = np.eye(3)
sigma[0,1] = 0.99
sigma[1,0] = 0.99

# #dumb fit
Xtr3 = np.random.multivariate_normal(means,sigma,size=N)
Ytr3 = np.dot(Xtr3,np.array([100,50,50]))
reg3 = LinearRegression(fit_intercept=False).fit(Xtr3, Ytr3)
explainer3 = shap.Explainer(reg3.predict,Xtr3)

names3 = ["X1","X2","X3"]
Xtr3_df = pd.DataFrame(Xtr3,columns=names3)

X5 = np.array([0.1,0.1,0.1]).reshape((1,3))
X5_df = pd.DataFrame(X5,columns=names3)

shap_values5 = explainer3(X5)
plot_rank(np.squeeze(shap_values5.values),Xtr3_df.columns.to_numpy(),"LinModOuts/m3e2_shap" )
delta_expl3 = DeltaExplainer(reg3,Xtr3_df,Nboot=500,saveDistr=True,useParallel=False)
res5 = delta_expl3.explain(X5_df,np.array([20]))
res5.plotRanking("LinModOuts/m3e2_delta")
res5.plotEffects("LinModOuts/m3e2_deltaE")


