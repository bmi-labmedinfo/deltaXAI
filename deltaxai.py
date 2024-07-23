from sklearn import *
from KDEpy import *
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import seaborn as sns
from fastkde import *
import time
from multiprocessing import  Pool
import logging


def evalPDFoutput_wrapper(instance, X, data_tmp, idx_feat,model_pred):
    """
    :param instance: instance of DeltaExplainer class
    :param X: current example to explain
    :param data_tmp: training set
    :param idx_feat: index of the current feature to explain
    :param model_pred: model prediction
    :return: pdf of model prediction given the conditioning con the single feature
    """
    return instance._evalPDFoutput(X, data_tmp, idx_feat,model_pred)


def getExplainKernels_wrapper(instance,X,data_tmp,idx_feat):
    return instance._getExplainKernels(X,data_tmp,idx_feat)

class pdf:
    """
    Class representing probability density functions
    """
    def __init__(self,samples):
        """
        :param samples: samples on which estimating the pdf
        """
        self.support, self.values = FFTKDE(kernel='gaussian', bw='silverman').fit(samples).evaluate()
    def evalPDF(self,point):
        """
        :param point: value at which pdf should be evaluated
        :return: pdf evaluate at the point
        """
        return self.values[np.argmin(abs((self.support - point)))]

    def evalPDFsupport(self,X):
        X = X.tolist()
        return np.array([self.evalPDF(pt) for pt in X])

    #write method to integrate pdf in a range

    def integrPDF(self,low,up):
        if low is None:
            idx = self.support<=up
        elif up is None:
            idx = self.support>=low
        else:
            idx = self.support>=low and self.support<= up

        xint = self.support[idx]
        yint = self.values[idx]
        return np.trapz(yint,xint).item()



class DeltaExplainerResult:

    """
    Class containing the methods and attributes describing the results of prediction explaination
    """

    def __init__(self,medians,low,high,directions,prediction,names,outName,X,y_true,decTh,kernels=None):

        """
        :param medians: median values of features indices
        :param low: 1st quartile of features indices
        :param high: 3rd quartile of features indices
        :param prediction: value of model prediction given X as input
        :param names: variable names
        :param outName: model output name
        :param X: example given as input
        :param y_true: true value
        """
        self.prediction = prediction
        self.medians = medians
        self.low = low
        self.high = high
        self.X = X
        self.varNames = names
        self.outName = outName
        self.__setRanking()
        self.y_true = y_true
        self.kernels = kernels
        self.decTh = decTh
        self.directions = directions

    @classmethod
    def initFromSim(cls,bootIdx,prediction,names,outName,X,y_true,decTh,kernels=None):
        """
        Alternative constructor to initialize Results given simulation outputs
        :param bootIdx: matrix containing the bootrapped values of the indices
        :param prediction: value of model prediction given X as input
        :param names: variable names
        :param outName: model output name
        :param X: example given as input
        :param y_true: true value
        :return: initialize DeltaExplainerResult object calling its constructor method
        """
        raw_signs = np.sign(bootIdx).sum(axis=0)
        Nboot = bootIdx.shape[0]

        directions = np.zeros(X.shape[1])
        directions[raw_signs > Nboot * 0.95] = 1
        directions[raw_signs < -Nboot*0.95] = -1

        bootIdx = abs(bootIdx)
        row_sum = bootIdx.sum(axis=1)
        row_sum = row_sum[:,np.newaxis]
        bootIdx = bootIdx/row_sum

        md = np.percentile(bootIdx,50,axis=0)
        idxs = md.argsort()
        idx_perc = bootIdx.argsort(axis=1)
        dif1 = idx_perc - idxs
        conc_median = np.sum(dif1 == 0,axis=1)

        if kernels is not None:
            kernels = kernels[conc_median.argmax()]

        return cls(
            medians=md,
            low=np.percentile(bootIdx,25,axis=0),
            high=np.percentile(bootIdx, 75, axis=0),
            directions= directions,
            prediction=prediction,
            names= names,
            outName=outName,
            X = X,
            y_true  =y_true,
            decTh = decTh,
            kernels = kernels
        )

    @classmethod

    def initFromCsv(cls,filename,prediction,outName,X,y_true,decTh):
        """
        Initialize DeltaExplainerResult from csv file
        :param filename: filename formatted as done from exportToCsv method from which loading info
        :param prediction: value of model prediction given X as input
        :param outName: model output name
        :param X: example given as input
        :param y_true: true value
        """
        df = pd.read_csv(filename,index_col=0)
        return cls(
            medians = df.loc["Median"].to_numpy(),
            low = df.loc["Low"].to_numpy(),
            high = df.loc["Up"].to_numpy(),
            directions= df.loc["Directions"].to_numpy(),
            prediction=prediction,
            names = df.columns,
            outName=outName,
            X = X,
            y_true =y_true,
            decTh=decTh,
            kernels = None
        )


    def plotRanking(self,figname,showDir=True,grouped=True):
        """
        Plot the ranking of the features
        :param figname: name of output file
        :return:
        """
        plt.rcParams.update({'font.size': 9})
        if grouped:
            colors = sns.color_palette("Paired", len(set(self.order)))
        else:
            colors = sns.color_palette("Paired", len(self.varNames))
        fig, ax = plt.subplots()
        bars = ax.barh(y=np.arange(len(self.medians)),
                width=self.medians[self.rank],
                xerr=np.array([self.medians[self.rank]-self.low[self.rank],
                               self.high[self.rank]-self.medians[self.rank]]),
                align="center",capsize=4)
        ax.set_xlim(None, self.high.max() * 1.1)
        ax.set_yticks(np.arange(len(self.medians)),self.__setTickFeatVals(self.varNames[self.rank]))
        ax.invert_yaxis()

        for i, bar in enumerate(bars):
            if grouped:
                bar.set_color(colors[self.order[i]-1])
            else:
                bar.set_color(colors[i])
            if showDir:
                ax.text(self.high[self.rank[i]]*1.02,
                        bar.get_y() +(bar.get_height()/2),
                        self.__dirToStr(self.directions[self.rank[i]].item()),
                        fontweight = "heavy",
                        verticalalignment="center")
            stop = 1
        truevalstr = " - true value: {0:.2f}".format(self.y_true.item())
        ax.set_title(self.__eqnToStr(self.outName,self.prediction.item())+truevalstr)
        plt.savefig(figname+".png", bbox_inches='tight', dpi=250)
        plt.clf()

    def __setRanking(self):
        """
        :return: initialize the ranking of the features as attributes within the class.
        """
        medians = self.medians
        rank_medians = medians.argsort()[::-1]

        self.rank = rank_medians
        self.order = []

        self.order.append(1)
        for i in range(1,len(rank_medians)):
            cm = medians[self.rank[i]]
            if cm < self.low[self.rank[i-1]]:
                self.order.append(self.order[i-1]+1)
            else:
                self.order.append(self.order[i - 1])

    def __dirToStr(self,d):
        if d==0:
            return "*"
        elif d>0:
            return"+"
        else:
            return"-"

    def __eqnToStr(self,what,value,sign="="):
        """
        Utility to write in plot equations
        :param what: variable
        :param value: var value
        :param sign: sign of the equation
        :return: formatted string equation
        """
        return what+" "+sign+" {0:.2f}".format(value)
    
    def __setTickFeatVals(self,names):
        """
        :param names: features names
        :return: list containing string representing feature values
        """
        return [self.__eqnToStr(z,self.X[[z]].to_numpy().item(),":") for z in names]


    def to_pandas(self):
        """
        :return: pandas dataframe summarizing the results of the explainer results
        """
        return pd.DataFrame(np.array([self.high[self.rank],
                       self.medians[self.rank],
                       self.low[self.rank],
                       self.order,
                      self.directions]),
                          columns=self.varNames[self.rank],
                            index=["Up","Median","Low","Order","Directions"]
                          )

    def exportToCsv(self,filename):
        """
        Export in a csv file explaination results
        :param filename: name of csv file
        :return:
        """
        self.to_pandas().to_csv(filename+".csv")

    def plotEffects(self,figname,topN=None):
        if self.kernels is None:
            raise Exception("No kernels stored. This method is NOT available when results are loaded from csv or when saveDistr is false in DeltaExplainer object")
        y = self.kernels[0]
        ycond = self.kernels[1:]

        if self.decTh is not None:
            self.__plotEffectsDecTh(figname,topN)
        else:
            plt.rcParams.update({'font.size': 14})
            fig, ax = plt.subplots(figsize=(19.2, 10.8))
            ax.plot(y.support,y.values,linewidth=2,label=self.outName)
            plot_order = self.rank
            MAX = y.values.max()
            if topN is not None:
                plot_order = plot_order[0:topN]
            for rank in plot_order:
                ax.plot(ycond[rank].support,ycond[rank].values,linewidth=2,
                        label=self.outName+" | "+self.varNames[rank]+" : "+
                              "{0:.2f}".format(self.X[[self.varNames[rank]]].to_numpy().item()))
                MAX = np.array([MAX,ycond[rank].values.max()]).max()
                stop = 1
            if self.decTh is None:
                ax.vlines(self.prediction,0,MAX,colors="k",linewidth=3,label=self.__eqnToStr(self.outName, self.prediction.item()))
            else:
                ax.vlines(self.decTh, 0, MAX, colors="k", linewidth=3,
                          label="Decisional threshold")
            ax.legend(loc="upper left")
            truevalstr = " - true value: {0:.2f}".format(self.y_true.item())
            ax.set_title(self.__eqnToStr(self.outName, self.prediction.item()) + truevalstr)
            if self.decTh is None:
                ax.set_xlabel(self.outName)
            else:
                ax.set_xlabel("p("+self.outName.split(" ")[-1][:-1]+") - dec. th {0:.2f}".format(self.decTh))
            ax.set_ylabel("pdf")
            ax.grid()
            plt.savefig(figname + ".png", bbox_inches='tight', dpi=600)
            plt.clf()

    def __plotEffectsDecTh(self,figname,topN):
        stop = 1
        plt.rcParams.update({'font.size': 9})
        fig, ax = plt.subplots(figsize=(19.2, 10.8))
        plot_order = self.rank
        if topN is not None:
            plot_order = plot_order[0:topN]
        base = self.kernels[0].integrPDF(self.decTh,None)
        kernels = self.kernels[1:]
        values = np.array([kernels[i].integrPDF(self.decTh,None) for i in plot_order])-base
        names = [self.varNames[i]+" : "+
                              "{0:.2f}".format(self.X[[self.varNames[i]]].to_numpy().item()) for i in plot_order]
        bars = ax.barh(y=np.arange(len(values)),
                       width=values)

        for i,bar in enumerate(bars):
            if values[i]>0:
                bar.set_color("orange")
        ax.set_yticks(np.arange(len(values)), names)
        ax.invert_yaxis()
        ax.axvline(x=0, color='r', label="p(" + self.outName.split(" ")[-1][:-1] + ") = {0:.2f}".format(base),linewidth=3)
        ax.set_title("p(" + self.outName.split(" ")[-1][:-1] + ") - dec. th {0:.2f}".format(self.decTh))
        ax.set_ylabel("Conditioning feature")
        ax.legend(loc='lower right')
        truevalstr = " - true value: {0:.2f}".format(self.y_true.item())
        ax.set_title(self.__eqnToStr(self.outName, self.prediction.item()) + truevalstr)
        plt.savefig(figname + ".png", bbox_inches='tight', dpi=250)
        plt.clf()


class DeltaExplainer:

    """"
    Class containing methods and attributes to explain model predictions
    """
    def __init__(self,model,training,cTarget=None,decTh=None,Nboot=500,seed=12345,useParallel=False,saveDistr=False):
        """
        :param model: model whose predictions should be explained
        :param training: training set
        :param cTarget: class of interest
        :param Nboot: Number of bootstrap samples to use when explaining features importance
        :param seed: seed to ensure reproducibility
        :param useParallel: whether parallelize computations over features
        """

        #parse inputs
        self.__parseInputs(cTarget,decTh)

        self.model = model
        self.training = training.to_numpy()
        self.cTarget = cTarget
        self.varNames = training.columns
        self.seed = seed
        self.Nboot = Nboot
        self.useParallel = useParallel
        self.saveDistr =  saveDistr
        self.decTh = decTh

    def __parseInputs(self,classT,thr):

        if classT is None:
            #regression problem
            if thr is not None:
                raise("If it is a regression problem, decisional threshold is not supported")

    def fitDistr(self,X):
        """
        Function to estimate the pdf of model output using the examples in X
        :param X: numpy array or pandas dataframe with N rows(=#examples) M columns (=#features)
        :return: pdf object
        """

        support_tmp = self.__getModelPrediction(X)

        kernel = pdf(support_tmp)
        return kernel


    def plotPDFfit(self,X,figname="checkPDFfit"):

        """
        Function to evaluate whether the fitted pdf is coherent with the observed samples
        :param X: examples on which pdf should be estimated
        :param figname: name of the output figure file
        """

        samples = self.__getModelPrediction(X)
        kernel = pdf(samples)
        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots()
        #fig.set_size_inches(32,18)
        ax.hist(samples, density=True,label="output")
        ax.plot(kernel.support, kernel.values, label='KDE',linewidth=3)

        if self.decTh is None:
            ax.set(
                xlabel= self.__getOutName(),
                ylabel="pdf"
            )
        else:
            ax.set(
                xlabel="p(" + self.__getOutName().split(" ")[-1][:-1] + ") - dec. th {0:.2f}".format(self.decTh),
                ylabel="pdf"
            )
        ax.legend(["hist. output","KDE"])
        ax.grid()
        plt.savefig(figname+".png", bbox_inches='tight', dpi=250)
        plt.clf()



    def explain(self,X,y_true,Xnat=None):

        """
        Function to explain features contribute on the prediction of a given instance
        :param X: Example to predict
        :param y_true: True value of the example
        :param Xnat: features de-normalized in natural scale
        :return: DeltaExplainerResult object
        """
        random.seed(self.seed)
        y_pred = self.__getModelPrediction(X.to_numpy())

        if self.decTh is not None and self.cTarget is not None:
            y_pred = self.__classifyPred(y_pred)
        kernel_boots = None
        if self.saveDistr:
            zz= [self.__bootIterSavingDistr(y_pred,X) for j in range(0,self.Nboot)]
            y_condf_PDF_boots = [el[0] for el in zz]
            kernel_boots = [el[1] for el in zz]
        else:
            y_condf_PDF_boots = [self.__bootIter(y_pred,X) for j in range(0,self.Nboot)]
        if Xnat is None:
            return DeltaExplainerResult.initFromSim(np.array(y_condf_PDF_boots), y_pred,
                                                    self.varNames, self.__getOutName(), X,y_true,self.decTh,kernel_boots)
        else:
            return DeltaExplainerResult.initFromSim(np.array(y_condf_PDF_boots),
                                                    y_pred,self.varNames,self.__getOutName(),Xnat,y_true,self.decTh,kernel_boots)
    def __bootIter(self,model_pred,Xsamp):

        """
        Function containing the operations performed on a bootstrap iteration when computning features importance
        :param model_pred: model prediction
        :param Xsamp: features of the examples to predict
        :return:
        """

        data = self.__getBootSamp()
        if self.useParallel:
            with Pool() as pool:
                args = [(self, np.squeeze(Xsamp.to_numpy()), np.copy(data), i,model_pred) for i in range(-1,Xsamp.shape[1])]
                feat_kernel_list = pool.starmap(evalPDFoutput_wrapper, args)
        else:
            feat_kernel_list = [self._evalPDFoutput(np.squeeze(Xsamp.to_numpy()), np.copy(data), i,model_pred) for i in
                                                  range(-1, Xsamp.shape[1])]
        delta_raw = np.array(feat_kernel_list[1:]) - feat_kernel_list[0]
        return delta_raw


    def __classifyPred(self,prob):
        if prob >= self.decTh:
            return np.array([self.cTarget])
        else:
            return np.array([int(not(self.cTarget))])



    def __bootIterSavingDistr(self,model_pred,Xsamp):
        data = self.__getBootSamp()
        if self.useParallel:
            with Pool() as pool:
                args = [(self, np.squeeze(Xsamp.to_numpy()), np.copy(data), i) for i in
                        range(-1, Xsamp.shape[1])]
                kernels = pool.starmap(getExplainKernels_wrapper,args)
        else:
            kernels = [self._getExplainKernels(np.squeeze(Xsamp.to_numpy()), np.copy(data), i) for i in
                                                  range(-1, Xsamp.shape[1])]
        if self.decTh is None:
            pdf_pred = [distr.evalPDF(model_pred) for distr in kernels]
        else:
            pdf_pred = [distr.integrPDF(self.decTh,None) for distr in kernels]
        delta_raw = np.array(pdf_pred[1:]) - pdf_pred[0]
        return delta_raw, kernels

    def __getOutName(self):
        """
        :return: String containing the output name
        """
        if self.cTarget is None:
            return "Model output"
        else:
            if self.decTh is None:
                return "".join(["p(class predicted = ", str(self.cTarget), " )"])
            else:
                return "Class Predicted (dec. th:"+"{0:.2f}".format(self.decTh)+" on class"+str(self.cTarget)+")"

    def __getBootSamp(self):
        """
        :return: extract a bootstrap (with replacement) sample of the training set
        """
        N,_ = self.training.shape
        idx_all =np.arange(0,N).tolist()
        idx = random.choices(idx_all,k=N)
        bootSamp = self.training[idx,:]
        return bootSamp

    def __getModelPrediction(self,X):
        """
        :param X: set of data to predict
        :return: model predictions
        """
        if self.cTarget is None:
            y = self.model.predict(X)
        else:
            y = self.model.predict_proba(X)
            y = y[:,self.cTarget]
        return y

    def _evalPDFoutput(self,X,data_tmp,idx_feat,model_pred):
        """
        :param X: features of the example
        :param data_tmp: sample of the training set
        :param idx_feat: index of the feature to explain. -1 is conventionally adopted to represent ignore contitional distr.
        :param model_pred: value of the model prediction when X is given as input
        :return: f(y) if idx_feat=-1, otherwise f(y|xi=x*)
        """
        if idx_feat>=0:
            data_tmp[:,idx_feat] = X[idx_feat]

        if self.decTh is None:
            return  self.fitDistr(data_tmp).evalPDF(model_pred)
        else:
            return self.fitDistr(data_tmp).integrPDF(self.decTh,None)


    def _getExplainKernels(self,X,data_tmp,idx_feat):
        if idx_feat>=0:
            data_tmp[:,idx_feat] = X[idx_feat]
        return  self.fitDistr(data_tmp)





