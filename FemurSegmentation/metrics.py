import itk
import numpy as np
from FemurSegmentation.utils import get_image_spatial_info, set_image_spatial_info


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



def dice_score(image1, image2) :
    '''
    '''
    y_true = itk.GetArrayFromImage(image1)
    y_pred = itk.GetArrayFromImage(image2)

    den = np.sum(y_true) + np.sum(y_pred)

    intersection = np.logical_and(y_true, y_pred)

    num = 2 * np.sum(intersection)
    return num / den



def housdorff_distance(image1, image2) :
    '''
    '''

    # FIXME I get "Image doesn't lie int same fisical space" even if is not so
    PixelType1, Dim1 = itk.template(image1)[1]
    PixelType2, Dim2 = itk.template(image2)[1]

    ImageType1 = itk.Image[PixelType1, Dim1]
    ImageType2 = itk.Image[PixelType2, Dim2]

    hd = itk.HausdorffDistanceImageFilter[ImageType1, ImageType2].New()
    _ = hd.SetInput1(image1)
    _ = hd.SetInput2(image2)
    _ = hd.Update()

    return hd.GetHausdorffDistance()
