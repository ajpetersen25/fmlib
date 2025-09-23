from fmlib.constants import *

### Standard imports
import os
import getpass
from datetime import datetime
import inspect

### SCIPY stack imports
import numpy as np
from scipy import interpolate

np.seterr(all='raise')
### Local PTV/PIV/IMG and HELPER modules
from fmlib.fio import ptvio
from fmlib.fio import imgio
#from fmlib.io import pivio

### Helper for profile code with line_profiler
# from profilecode import profile

# if __debug__:>
#     print 'Debug is ON'

######################################################################
# ptvbase - skeleton class with standard methods
#
# This ONLY serves as a subclass. Place any generic functions in here
# that have common algorithms
######################################################################
class ptvbase:
    '''
    Base class of ptv methods
    This class cannot be initiated, but serves as a base class
    with default methods that can be used by subclasses
    '''

    def __init__(init,**kwargs):
        '''
        This should not be called, as all calls should be to a subclass
        '''
        raise Exception("Cannot initiate base class ptvcrossbase. Please specify a valid class" )

    def _init_cross_method(self,cross_method):
        '''
        Store method of ptv in class instance

        Arguments:
          method - str - Valid cross-correlation method of calling class

        Returns:
          None
        '''

        self.cross_method = cross_method

    def _init_vars(self, start_frame, track_increment, no_of_pairs, pair_increment, threshold,
                   lowpass, highpass, peak_method, radius, search_radius, snrvalue, centroid_method,
                   n_cores, lsq_bounds):
        '''
        Initialise instance variables for all particle tracking

        Arguments:
          same as class

        Returns:
          None
        '''
        # Variable allocation
        self.start_frame = start_frame
        self.no_of_pairs = no_of_pairs
        self.pair_increment = pair_increment
        self.track_increment = track_increment
        self.srs = search_radius**2

        self.search_radius= search_radius
        self.snrvalue = snrvalue

        # peak finding and bandpass filtering TODO, strict and crop autoset
        self.lowpass = lowpass
        self.highpass = highpass
        self.peak_method = peak_method

        self.radius= radius

        self.crop = radius

        self.threshold = threshold # only a border of 1 is needed

        # TODO multipass in future.
        self.passes = 1

        self.centroid_method = centroid_method
        self.n_cores = n_cores
        # can only do lsq if CoM used
#        if self.peak_method == 'com':
#            if self.centroid_method != 'lsq':
#                print '[WARNING] Centroid peak detection only works with lsq subpixel'
#            self.centroid_method = 'lsq'

        self.lsq_bounds = lsq_bounds


    def _init_img(self,img_filename):
        '''
        Create and assign an imgio object based on img filename to class instance
        Store file_root as class instance variable

        Arguments:
          img_filename - str - Valid relative path to IMG file

        Returns:
          None
        '''

        # intialise img object
        self.img = imgio.imgio(img_filename)

        # separate out IMG file and extension
        self.file_root,_ = os.path.splitext(self.img.file_name)

    def _init_frame_index(self):
        '''
        Generates and assigns list of frameA indices to process to instance variables
        Assigns the number of frames in final PTV

        Arguments:
          None

        Returns:
          None
        '''
        # generate list of frameA indices
        self.frames_to_process = np.arange(self.start_frame,
                                           self.no_of_pairs*self.pair_increment+self.start_frame,
                                           self.pair_increment)

        self.nt = len(self.frames_to_process)

    def _init_mask(self):
        '''
        Looks for a .msk file (renamed .img file) and assigns it

        Arguments:
          None

        Returns:
          None
        '''

        # mask filename
        mask_file = '%s.msk' % (self.file_root)

        # test for mask
        if os.path.exists(mask_file) == 0:
            print('[GENERAL] No mask file. Processing all vectors')
        else:
            # intialise img object
            maskimg = imgio.imgio(mask_file)
            self.mask = maskimg.read_frame2d(0)

            print('[GENERAL] Mask file found.')

    def print_status(self,status):
        '''
        Takes a status vector and prints to screen the sum of the vectors

        Arguments:
          status - 1D array of vector statuses

        Returns:
          None
        '''

        print('On : %s\tNo Sub : %s\tOut : %s\tFilled : %s\tTotal : %s' % ('{:>5}'.format((status==1).sum()),
                                                                           '{:>5}'.format((status==3).sum()),
                                                                           '{:>5}'.format((status==4).sum()),
                                                                           '{:>5}'.format((status==2).sum()),
                                                                           '{:>5}'.format((status<5).sum())))
        print()
