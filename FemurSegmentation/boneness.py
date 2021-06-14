import numpy as np

from FemurSegmentation.utils import image2array
from FemurSegmentation.utils import array2image
from FemurSegmentation.utils import get_image_spatial_info

from FemurSegmentation.filters import gaussian_smoothing
from FemurSegmentation.filters import hessian_matrix
from FemurSegmentation.filters import get_eigenvalues_map
from FemurSegmentation.filters import execute_pipeline

# TODO add healt check and error/exception handling


class Boneness :

    def __init__(self, image, scales=[0.5], roi=None) :
        '''
        Class to compute the boneness measure of the filter

        Parameters
        ----------
        image : itk.Image
            image from which compute the boneness measure
        scales : list of float
            list of scales used to compute the filter
        roi : itk.Image
            if specified, the filter will be computed only on the roi region.
            Otherwise will be empty.
        '''

        if roi is not None :
            r, _ = image2array(roi)
        else:
            r = roi

        self.roi = r
        self.image = image
        self.scales = scales

    def computeEigenvaluesMeasures(self, eigenvalues_map) :
        '''
        '''
        eigen, _ = image2array(eigenvalues_map)

        # I'm computing the eigenvalues absolute value and also taking only the
        # real eignevalues (the first three element of the tensor), excluding in
        # his way th eigenvectors (??) -> chek in the docs to be shure!!
        #
        # After that i will get all the matrix element with the laregerst
        # eigenvalues (the third one) iferent from zero, that because it is at
        # the denominator in the computing of one particular quantity
        eigen_abs = np.abs(eigen[:, :, :, :3])
        eigen_no_null = ~np.isclose(eigen_abs[:, :, :, 2], 0)

        # strat the computing of he eigen quantity for the estimation of the
        # boneness
        R_bones = np.empty(eigen_abs.shape[:-1], dtype=np.float32)

        det_image = np.sum(eigen_abs, axis=-1)

        if self.roi is not None :
            # compute the mean only inside the region of interest
            mean_norm = 1. / np.mean(det_image[self.roi != 0])
        else :
            mean_norm = 1. / np.mean(det_image)
        R_noise = det_image * mean_norm

        R_bones[eigen_no_null] = (eigen_abs[eigen_no_null, 0] * eigen_abs[eigen_no_null, 1]) / eigen_abs[eigen_no_null, 2] ** 2

        # FIXME I do not preserve image informations!!
        return R_bones, R_noise, eigen_no_null, eigen

    def bonenessMeasure(self, scale, alpha=0.5, beta=0.05) :
        '''
        '''
        # pipeline to get the eigenvalues map:
        sm = gaussian_smoothing(self.image, sigma=scale)
        hess = hessian_matrix(sm.GetOutput(), sigma=scale)
        pipe = get_eigenvalues_map(hess.GetOutput())

        eigen_map = execute_pipeline(pipe)

        R_bones, R_noise, eigen_no_null, eigen = self.computeEigenvaluesMeasures(eigen_map)

        R_s = R_bones**2
        measure = np.empty(R_bones.shape)

        measure[eigen_no_null] = - np.sign(eigen[eigen_no_null, 2])
        measure[eigen_no_null] *= np.exp(- 0.5 * R_s[eigen_no_null])
        measure[eigen_no_null] *= (1 - np.exp(- R_noise[eigen_no_null]
                                            * R_noise[eigen_no_null] * 4))

        return measure

    def computeBonenessMeasure(self, alpha=0.5, beta=0.05) :
        '''
        '''
        ms_measure = self.bonenessMeasure(self.scales[0])

        for scale in self.scales[1:] :
            tmp = self.bonenessMeasure(scale)
            cond = np.abs(tmp) > np.abs(ms_measure)
            ms_measure[cond] = tmp[cond]
        info = get_image_spatial_info(self.image)
        bones = array2image(ms_measure, info)

        return bones
