import numpy as np

from FemurSegmentation.utils import image2array
# TODO add healt check and error/exception handling

__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']


class GraphCutLinks :

    def __init__(self,
                image,
                boneness,
                roi,
                obj,
                bkg,
                sigma=.25,
                Lambda=50.,
                bone_ms_thr=0.2) :
        '''
        '''

        # TODO add input controls
        self.image, _ = image2array(image)
        self.boneness, _ = image2array(boneness)
        self.roi, _ = image2array(roi)
        self.bkg, _ = image2array(bkg)
        self.obj, _ = image2array(obj)
        self.total_vx = self.roi[self.roi == 1].size#int(np.sum(self.roi[self.roi == 1]))
        self.vx_id = np.full(self.image.shape, -1)
        self.vx_id[self.roi != 0] = range(self.total_vx)

        self.sigma = sigma
        self.Lambda = float(Lambda)
        self.bone_ms_thr = bone_ms_thr

    def bonenessCost(self, vx_left, vx_right, sh_left, sh_right) :
        '''
        '''
        # both voxel must be in ROI
        cond = (vx_left > -1) & (vx_right > -1)

        from_center = np.full(vx_left[cond].shape, self.Lambda)
        to_center = np.full(vx_left[cond].shape, self.Lambda)

        den = 2. * (self.sigma ** 2)
        num = np.abs(sh_left[cond] - sh_right[cond])**2

        # compute cost from center
        cond_a = sh_left[cond] > sh_right[cond]
        from_center[cond_a] *= np.exp(- num[cond_a] / den)

        # compute cost to center
        cond_b = sh_left[cond] < sh_right[cond]
        to_center[cond_b] *= np.exp(- num[cond_b] / den)

        return (vx_left[cond], vx_right[cond]), to_center, from_center

    def tLinkSource(self) :
        cost_source = np.zeros(self.image.shape)
        # voxel belong  to source (obj) and is in ROI
        cond = (self.obj == 1) & (self.roi == 1)
        cost_source[cond] = self.Lambda

        cond = (self.obj == 0) & (self.roi == 1) & (self.boneness > self.bone_ms_thr) & (self.bkg == 0)
        cost_source[cond] = 1

        return cost_source

    def tLinkSink(self) :
        cost_sink = np.ones(self.image.shape)

        cond = (self.bkg == 1) & (self.roi == 1)
        cost_sink[cond] = self.Lambda

        cond = (self.obj == 1) & (self.roi == 1)
        cost_sink[cond] = 0

        return cost_sink

    def nLinks(self) :

        X, Xto, Xfrom = self.bonenessCost(self.vx_id[:-1, :, :],
                                          self.vx_id[1:, :, :],
                                          self.boneness[:-1, :, :],
                                          self.boneness[1:, :, :])

        Y, Yto, Yfrom = self.bonenessCost(self.vx_id[:, :-1, :],
                                          self.vx_id[:, 1:, :],
                                          self.boneness[:, :-1, :],
                                          self.boneness[:, 1:, :])

        Z, Zto, Zfrom = self.bonenessCost(self.vx_id[:, :, :-1],
                                          self.vx_id[:, :, 1:],
                                          self.boneness[:, :, :-1],
                                          self.boneness[:, :, 1:])

        CentersVx = np.concatenate([Z[0], Y[0], X[0]])
        NeighborsVx = np.concatenate([Z[1], Y[1], X[1]])
        _totalNeighbors = len(NeighborsVx)
        costFromCenter = np.concatenate([Zfrom, Yfrom, Xfrom])
        costToCenter = np.concatenate([Zto, Yto, Xto])

        return CentersVx, NeighborsVx, _totalNeighbors, costFromCenter, costToCenter

    def getLinks(self) :

        # get tLinks
        source = self.tLinkSource()
        sink = self.tLinkSink()
        # flatten tLinks
        cost_sink_flatten = sink[self.vx_id != -1]
        cost_source_flatten = source[self.vx_id != -1]
        cost_vx = self.vx_id[self.vx_id != -1]

        # nLinks
        CentersVx, NeighborsVx, _totalNeighbors, costFromCenter, costToCenter = self.nLinks()

        return cost_sink_flatten, cost_source_flatten, cost_vx, CentersVx, NeighborsVx, _totalNeighbors, costFromCenter, costToCenter
