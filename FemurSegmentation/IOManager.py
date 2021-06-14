import os
import itk
import numpy as np

__author__ = ['Biondi Riccardo']
__email__ = ['riccardo.biondi7@unibo.it']


# ██████  ███████  █████  ██████  ███████ ██████
# ██   ██ ██      ██   ██ ██   ██ ██      ██   ██
# ██████  █████   ███████ ██   ██ █████   ██████
# ██   ██ ██      ██   ██ ██   ██ ██      ██   ██
# ██   ██ ███████ ██   ██ ██████  ███████ ██   ██

# TODO add healt check and error/exception handling


class ImageReader :
    '''
    Read a medical image (nrrd, nifty, etc) or a dicom series and return an itk
    Image object

    Attributes
    ----------
    path : str
        path to image file or folder containing the DICOM series
    image_type : itk image type
        input image type, i.e. itk.Image[itk.UC, 3]

    Methods
    -------

    Example
    -------
    >>> import IOManager as iom
    >>> import itk
    >>> path='./path/to/image/file.nrrd'
    >>>
    >>> ImageType = itk.Image[itk.UC, 3]
    >>>
    >>> reader = iom.ImageReader(path = path, image_type = ImageType)
    >>> image = reader.read()
    '''

    def __init__(self, path="", image_type=itk.Image[itk.UC, 3]):
        '''
        Parameters
        ----------
        path : str

        image_type : itk Image obj

        '''
        self.path = path
        self.image_type = image_type

    def __call__(self, path, image_type=itk.Image[itk.UC, 3]) :
        self.path = path
        self.image_type = image_type
        return self.read()

    def isPath2File(self) :
        '''
        Check if the path is a path to a directory wih contains a DICOM series or
        to file in medical image format.

        Return
        ------
        True : if the path is to a image file

        False : if path is to a folder containing the DICOM series

        Raise
        -----
        OSError : if the path does not exist
        '''

        if os.path.exists(self.path) :
            if os.path.isfile(self.path) :
                return True
            else :
                return False
        else :
            raise OSError("The specified path or image file : %s \
                            does not exists" % self.path)

    def DICOM2Volume(self) :
        '''
        Will read a DICOM series and convert it into an image tensor
        '''
        seriesGenerator = itk.GDCMSeriesFileNames.New()
        seriesGenerator.SetUseSeriesDetails(True)  # Use images metadata
        seriesGenerator.AddSeriesRestriction("0008|0021")  # Series Date
        seriesGenerator.SetGlobalWarningDisplay(False)  # disable warnings
        seriesGenerator.SetDirectory(self.path)

        # Get all indexes serieses and keep the longest series
        # (in doing so, we overcome the issue regarding the first CT sampling scan)
        seqIds = seriesGenerator.GetSeriesUIDs()
        UIDsFileNames = [seriesGenerator.GetFileNames(seqId) for seqId in seqIds]
        LargestDicomSetUID = np.argmax(list(map(len, UIDsFileNames)))
        LargestDicomSetFileNames = UIDsFileNames[LargestDicomSetUID]

        dicom_reader = itk.GDCMImageIO.New()
        reader = itk.ImageSeriesReader[self.image_type].New()
        reader.SetImageIO(dicom_reader)
        reader.SetFileNames(LargestDicomSetFileNames)
        # Since CT acquisition is not fully orthogonal (gantry tilt)
        _ = reader.ForceOrthogonalDirectionOff()
        _ = reader.Update()

        return reader.GetOutput()

    def image2Volume(self) :
        '''
        Read a Medical image as itk Image volume
        '''
        reader = itk.ImageFileReader[self.image_type].New()
        _ = reader.SetFileName(self.path)
        _ = reader.Update()

        return reader.GetOutput()

    def read(self) :
        '''
        '''
        if self.isPath2File() :
            return self.image2Volume()
        else :
            return self.DICOM2Volume()

            # ██     ██ ██████  ██ ████████ ███████ ██████
            # ██     ██ ██   ██ ██    ██    ██      ██   ██
            # ██  █  ██ ██████  ██    ██    █████   ██████
            # ██ ███ ██ ██   ██ ██    ██    ██      ██   ██
            #  ███ ███  ██   ██ ██    ██    ███████ ██   ██


class VolumeWriter :
    '''
    Write and ITK image as medical image format(i.e. nrrd, nifti) or DICOM series

    Attributes
    ----------

    path : str
        output path (if DICOM) or output image name
    image : itk Image
        volume to write
    as_dicom : bool
        if True, write image as DICOM series

    Example
    -------
    >>> import IOManager as iom
    >>> import itk
    >>>
    >>> in_path = '/path/to/input/image'
    >>> in_type = itk.Image[itk.UC, 3]
    >>>
    >>> reader = iom.ImageReader(path = in_path, image_type = in_type)
    >>> image = reader.read()
    >>>
    >>> # Process Image
    >>> out_path = '/path/to/out/folder/filaname.nrrd'
    >>> # Write volume as .nrrd
    >>> wirter = iom.VolumeWriter(path = out_path, image = image)
    >>> writer.write()
    '''

    def __init__(self, path="", image=None, as_dicom=False) :
        self.path = path
        self.image = image
        self.as_dicom = as_dicom

    def __call__ (self, path, image, as_dicom=False) :
        self.path = path
        self.image = image
        self.as_dicom = as_dicom

        self.write()

    def GetImageType(self) :
        PixelType, Dimension = itk.template(self.image)[1]

        return itk.Image[PixelType, Dimension]

    def volume2DICOM(self) :
        '''
        Write itk image as DICOM series
        '''
        # of not exists, create the output directory
        _ = os.makedirs(self.path, exist_ok=True)
        slice_format = os.path.join(self.path, "%03d.dcm")
        largest_region = self.image.GetLargestPossibleRegion()

        # generate the series of dicom images (one for each volume slice)
        series = itk.NumericSeriesFileNames.New()
        _ = series.SetSeriesFormat(slice_format)
        _ = series.SetStartIndex(largest_region.GetIndex()[2])
        _ = series.SetEndIndex(largest_region.GetIndex()[2] +
                            largest_region.GetSize()[2] - 1)
        _ = series.SetIncrementIndex(1)

        # prepare the writer object and write the series
        PixelType, _ = itk.template(self.image)[1]
        dicom_io = itk.GDCMImageIO.New()
        input_type = itk.Image[PixelType, 3]  # takes a volume as input
        output_type = itk.Image[PixelType, 2]  # write a series of 2D iamges

        writer = itk.ImageSeriesWriter[input_type, output_type].New()
        _ = writer.SetInput(self.image)
        _ = writer.SetImageIO(dicom_io)
        _ = writer.SetFileNames(series.GetFileNames())

        _ = writer.Update()

    def volume2Image(self) :
        '''
        Write and itk image obj in a medical image format
        '''
        writer = itk.ImageFileWriter[self.GetImageType()].New()
        _ = writer.SetFileName(self.path)
        _ = writer.SetInput(self.image)
        _ = writer.Update()

    def write(self) :
        '''
        '''
        if self.as_dicom :
            self.volume2DICOM()
        else :
            self.volume2Image()
