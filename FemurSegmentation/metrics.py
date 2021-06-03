import itk
import numpy as np


__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']

# TODO add healt check and error/exception handling


class SimilarityMeasures :
    '''
    '''

    def __init__ (self, y_pred, y_true) :

        if ~isinstance(y_pred, np.array) :
            self.y_pred = itk.GetArrayFromImage(y_pred)
        else :
            self.y_pred = y_pred
        if  ~isinstance(y_true, np.array) :
            self.y_true = itk.GetArrayFromImage(y_true)
        else :
            self.y_true = y_true

        # compute the confusion matrix ??
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        pass

    def confusionMatrix(self) :
        pass


    def diceScore(self) :
        pass

    def jaccardIndex(self) :
        pass

    def sensitivity(self) :
        pass

    def sepecificity(self) :
        pass

    def globalConsistencyError(self) :
        pass

    def volumeSimilarity(self) :
        pass

    def interclassCorrelation(self) :
        pass

    def probabilisticDistance(self) :
        pass


    def getSimilarityMeasuresDict(self) :
        '''
        '''

        metrics = {
                'Dice Score' : self.diceScore(),
                'Jaccard Index' : self.jaccardIndex(),
                'Sensitivity' : self.sensitivity(),
                'Speificity' : self.sepecificity(),
                'Volume Similarity' : self.volumeSimilarity(),
                'Interclass Correlation' : self.interclassCorrelation(),
                'probabilisticDistance' : self.probabilisticDistance()
        }

        return metrics
