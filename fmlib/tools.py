######################################################################
# DEPENDENCIES
######################################################################
import sys
import numpy as np
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
import scipy.ndimage.measurements as measurements
from scipy.optimize import curve_fit
from scipy.special import erf

# import constants
from fmlib.constants import *

# used by registration_array_3ptregistration
NL = np.log1p(np.arange(0,2**16))

import matplotlib.pyplot as plt

def bandpass(img,lowpass,highpass=0,threshold = 0):
    """
    Band pass filter using Gaussian and Boxcar

    Usage:

    bandpass(img,lowpass,highpass,threshold)

    Arguments
      img       - grayscale image matrix
      lowpass   - length scale of noise
      highpass  - radius of particles
      threshold - threshold out particles - off by default.
    """

    img2 = np.asarray(img,dtype=np.float)
    img2[img2 < threshold] = 0;
    gimg = filters.gaussian_filter(img2,lowpass,mode='constant',cval = 0)

    if highpass > 0:

        bimg = filters.uniform_filter(img2,highpass,mode='nearest',cval=0)
        gimg -= np.abs(bimg)

    gimg[gimg<0] = 0

    return gimg.astype(img.dtype)

def threshold(img,radius,threshold):
    data_max = filters.maximum_filter(img, radius)

    maxima = (img == data_max)
    data_min = filters.minimum_filter(img, radius)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    return maxima

def peak_finder_func(peak_method, img, radius,threshold=0,crop=None):
    '''
    peak_finder_func returns the pixels of the centroids in labelled images based on function

    Usage: peak_finder_method(function, img, radius, threshold)

    Arguments
      img       - np.array dtype=int   image array
      radius    - expected radius - larger features are ignored
      threshold - intensities smaller than this value

    Returns
      xy        - nx2 numpy array of x,y locations
    '''
    if crop is None:
        crop = radius

    #print 'Peak Finder Func Settings'
    #print peak_method
    #print radius
    #print threshold
    #print crop

    xy = getattr(sys.modules[__name__],'peak_finder_%s'% peak_method)(img.copy(), radius,threshold=threshold)

    edges = (xy[:,0] < crop) | (xy[:,0] > (img.shape[1] - crop - 1)) | (xy[:,1] < crop) | (xy[:,1] > (img.shape[0] - crop - 1))
    return xy[~edges]

def peak_finder_com(img, radius,threshold=0):
    '''
    peak_finder_CoM returns the pixels of the centroids in labelled images

    Usage: peak_finder_CoM(img, radius, threshold)

    Arguments
      img       - np.array dtype=int   image array
      radius    - expected radius - larger features are ignored
      threshold - intensities smaller than this value

    Returns
      xy        - nx2 numpy array of x,y locations
    '''

    img[img < threshold] = 0

    lab, num_obj = measurements.label(img)
    slices = measurements.find_objects(lab)
    xy = np.zeros((len(slices),2))
    f2 = np.copy(img)
    for i, dxy in enumerate(slices):

        slab = lab[dxy]
        simg = np.copy(f2[dxy])
        simg[slab!=(i+1)] = 0
        c = measurements.center_of_mass(simg)
        xy[i,:] = dxy[1].start+np.round(c[1]),dxy[0].start+np.round(c[0])

    return xy

def peak_finder_com_subpixel(img, radius,threshold=0):
    '''
    peak_finder_CoM returns the pixels of the centroids in labelled images

    Usage: peak_finder_CoM(img, radius, threshold)

    Arguments
      img       - np.array dtype=int   image array
      radius    - expected radius - larger features are ignored
      threshold - intensities smaller than this value

    Returns
      xy        - nx2 numpy array of x,y locations
    '''

    img[img < threshold] = 0

    lab, num_obj = measurements.label(img)
    slices = measurements.find_objects(lab)
    xy = np.zeros((len(slices),2))
    f2 = np.copy(img)
    for i, dxy in enumerate(slices):

        slab = lab[dxy]
        simg = np.copy(f2[dxy])
        simg[slab!=(i+1)] = 0
        c = measurements.center_of_mass(simg)
        xy[i,:] = dxy[1].start+c[1],dxy[0].start+c[0]

    return xy

def peak_finder_strict(img,radius=None,threshold=0):
    '''
    peak_finder_LocalMaxima returns the pixels of local maxima

    Usage: peak_finder_LocalMaxima(img, radius, threshold)

    Arguments
      img       - np.array dtype=int   image array
      radius    - ignored
      threshold - intensities smaller than this value

    Returns
      xy        - nx2 numpy array of x,y locations
    '''

    y,x = np.where((img >= threshold) & (img > np.roll(img,1,0)) & (img > np.roll(img,-1,0)) & (img > np.roll(img,1,1)) & (img > np.roll(img,-1,1)))
    xy = np.zeros((len(y),2),np.int)
    xy[:,0] = x
    xy[:,1] = y

    return xy

def peak_finder_maxfilter(img,radius,threshold=0):
    '''
    peak_finder_MaximumFilter returns the pixels of local maxima

    Usage: peak_finder_MaximumFilter(img, radius, threshold)

    Arguments
      img       - np.array dtype=int   image array
      radius    - expected radius - larger features are ignored
      threshold - intensities smaller than this value

    Returns
      xy        - nx2 numpy array of x,y locations
    '''

    data_max = filters.maximum_filter(img, radius)
    maxima = (img == data_max)
    data_min = filters.minimum_filter(img, radius)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    xy = np.zeros((len(slices),2),np.int)
    img2 = np.copy(img)
    for i,dxy in enumerate(slices):
        slab = labeled[dxy]
        simg = np.copy(img2[dxy])
        simg[slab!=(i+1)] = 0
        y,x = np.unravel_index(np.argmax(simg),simg.shape)
        xy[i,:] =  x+dxy[1].start, y+dxy[0].start

    return xy

def registration_array(centroid_method,img,xy,radius=PARTICLE_RADIUS,lsq_bounds=PARTICLE_LSQ_BOUNDS):
    '''
    Returns gaussian registration of peaks in a scalar field (img) around maxima given by a 2 column x,y array (xy)

    Usage:
        registration_array_3pt(img,xy)

    Arguments
        img    - 2D int array (currently 2-16bit INTEGER)
        xy     - 2D numpy array of rows of [x,y] locations on which to fit gaussian

    Returns
        xys    - 2D numpy array of x,y

    '''
    return getattr(sys.modules[__name__],'registration_array_%s' % centroid_method)(img.astype(np.int), xy.astype(np.int),radius,lsq_bounds)


def registration_array_3pt(img,xy,radius=PARTICLE_RADIUS,lsq_bounds=PARTICLE_LSQ_BOUNDS):
    '''
    Returns gaussian registration of peaks in a scalar field (img) around maxima given by a 2 column x,y array (xy)

    Usage:
        registration_array_3pt(img,xy)

    Arguments
        img    - 2D int array (currently 2-16bit INTEGER)
        xy     - 2D numpy array of rows of [x,y] locations on which to fit gaussian

    Returns
        xys    - 2D numpy array of x,y

    '''

    sxy = np.zeros((xy.shape[0],2))
    sxy[:,0:2] = xy

    dx = (NL[img[xy[:,1],xy[:,0]-1]]+NL[img[xy[:,1],xy[:,0]+1]]-2*NL[img[xy[:,1],xy[:,0]]])
    dy = (NL[img[xy[:,1]-1,xy[:,0]]]+NL[img[xy[:,1]+1,xy[:,0]]]-2*NL[img[xy[:,1],xy[:,0]]])

    sx = np.abs(dx)>EPSILON
    sy = np.abs(dy)>EPSILON

    sxy[sx,0] += 0.5*(NL[img[xy[sx,1],xy[sx,0]-1]]-NL[img[xy[sx,1],xy[sx,0]+1]])/dx[sx]
    sxy[sy,1] += 0.5*(NL[img[xy[sy,1]-1,xy[sy,0]]]-NL[img[xy[sy,1]+1,xy[sy,0]]])/dy[sy]

    return sxy

def registration_array_lsq(img,xy,radius=PARTICLE_RADIUS,lsq_bounds=PARTICLE_LSQ_BOUNDS):
    c = np.zeros(xy.shape)
    for i in np.arange(xy.shape[0]):

        simg = img[(-radius+xy[i,1]):(radius+xy[i,1]+1),(-radius+xy[i,0]):(radius+xy[i,0]+1)]

        try:
            c[i,:], cv = gaussian_fit(simg,2*radius,lsq_bounds)
        except:
            try:
                simg_c = measurements.center_of_mass(simg)
                c[i,:] = simg_c[1],simg_c[0]
            except:
                c[i,:] = radius
    return c+xy-radius

def gen_gaussian2d(xy,amp,x0,y0,radius):
    x,y=xy
    Hx=0.5*(erf(((x-x0)+radius)/(np.sqrt(2)*radius*.5))-erf(((x-x0)-radius)/(np.sqrt(2)*radius*.5)))
    Hy=0.5*(erf(((y-y0)+radius)/(np.sqrt(2)*radius*.5))-erf(((y-y0)-radius)/(np.sqrt(2)*radius*.5)))
    return amp*Hx*Hy

def gaussian_fit(img, radius=2,lsq_bounds=3):
    """
    function to fit a 2D (symmetric) gaussian using non-linear least-squares fitting

    Usage:

    Arguments:
        img    - 2D int array
        xy     - 2D numpy array of rows of [x,y] locations on which to fit gaussian
        radius - integer guess of gaussian looking

    Returns:
       centroid         -array  - predicted centroid in z (col,row)
       centroid_var     -array - variance in predicted centroid
    """

    guess_params = [np.max(img),img.shape[1]//2,img.shape[0]//2,radius]

    x=np.arange(0,img.shape[1])
    y=np.arange(0,img.shape[0])

    xi,yi = np.meshgrid(x,y,indexing='xy')
    xyi = np.vstack([xi.ravel(),yi.ravel()])
    values = img.flatten()
    pred_params, uncert_cov = curve_fit(gen_gaussian2d,xyi,values,guess_params,bounds=([1,img.shape[1]//2-lsq_bounds,img.shape[0]//2-lsq_bounds,.5*radius],[np.inf,img.shape[1]//2+lsq_bounds,img.shape[0]//2+lsq_bounds,3*radius]))
    centroid = np.array([pred_params[1],pred_params[2]])
    centroid_var = np.array([uncert_cov[1,1],uncert_cov[2,2]])
    return centroid,centroid_var#,pred_params


#############################################################
# Subpixel registration methods for correlation maps
#############################################################
def subpixel_registration_snr(smap,snrvalue,method='gaussian'):
    '''
    Takes a centered correlation map
    Returns the subpixel location (x,y) of the maximum peak
    depending on different method

    Arguments:
      smap     - 2D array correlation map
      snrvalue - float of the SNR critertion: second peak must be less than 1/snrvalue of first
      method   - str - method of registration default='gaussian','polynomial'

    Returns value from piv registration methods (see piv.py):
      status - int - 1 for success, 3 for no success
      xi     - float of column location of peak
      yi     - float of row location of peak
      snr    - bool SNR test pass/fail

    '''
    d = np.nanargmax(smap)
    i = d%smap.shape[1]
    j = d//smap.shape[1]
    smap = smap-np.min(smap.flatten())
    mmax= smap.flatten()[d]

    if mmax>0:
        smap = smap/mmax

        status,xi,yi= getattr(sys.modules[__name__],'%s_registration' % method)(smap[j-1:j+2,i-1:i+2])

        xi = xi+i-smap.shape[1]//2
        yi = yi+j-smap.shape[0]//2

        snr = 1
        smap[j-1:j+2,i-1:i+2]=0

        if (smap > 1./snrvalue).sum() > 0:
            snr = 0
            status = 2
    else:
        snr = 0
        status = 3
        xi = i-smap.shape[1]//2
        yi = j-smap.shape[0]//2


    return status,xi,yi,snr

def gaussian_registration(smap):
    '''
    Returns the subpixel peak centered (0,0) using 3 point 1D-gaussian peak fitting
    Only works on integers

    Arguments:
       smap - 2D matrix of centered correlation map - 2-16bit integer accepted

    Returns
       xi   - float of column location of peak
       yi   - float of row location of peak

    '''

    if True:

        xi = 0.5*(np.log1p(smap[1,0])-np.log1p(smap[1,2]))/(np.log1p(smap[1,0])+np.log1p(smap[1,2])-2*np.log1p(smap[1,1]))
        yi = 0.5*(np.log1p(smap[0,1])-np.log1p(smap[2,1]))/(np.log1p(smap[0,1])+np.log1p(smap[2,1])-2*np.log1p(smap[1,1]))

        status = 1
    else:

        status = 3
        xi = 0
        yi = 0

    return status,xi,yi


def polynomial_registration(smap):
    '''
    Returns the polynomial maxima around the maximum integer value based on a 3x3 stencil centered on maxima (0,0)

    Arguments:
       smap - 2D matrix
       recursion - allows a recursion on the image with a gaussian

    Returns
       status - 1 for success, 3 for no success
       xi   - float of column location of peak
       yi   - float of row location of peak

    '''

    try:
        a = (smap[1,0]+smap[0,0]-2*smap[0,1]+smap[0,2]-2*smap[2,1]-2*smap[1,1]+smap[1,2]+smap[2,0]+smap[2,2])
        b = (smap[2,2]+smap[0,0]-smap[0,2]-smap[2,0])
        c = (-smap[0,0]+smap[0,2]-smap[1,0]+smap[1,2]-smap[2,0]+smap[2,2])
        e = (-2*smap[1,0]+smap[0,0]+smap[0,1]+smap[0,2]+smap[2,1]-2*smap[1,1]-2*smap[1,2]+smap[2,0]+smap[2,2])
        f = (-smap[0,0]-smap[0,1]-smap[0,2]+smap[2,0]+smap[2,1]+smap[2,2])
        yi = (6.*b*c-8*a*f)/(16*e*a-9*b**2)
        xi = (6.*b*f-8*e*c)/(16*e*a-9*b**2)
        status = 1
    except:
        status = 3
        xi = 0
        yi = 0

    return status, xi, yi


