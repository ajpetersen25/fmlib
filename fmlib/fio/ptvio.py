import math
import numpy as np
import shutil
#from IPython.Debugger import Tracer; debug_here = Tracer()
from numpy.lib.stride_tricks import as_strided
import code
import getpass
import h5py
import os,sys
import scipy.ndimage.filters as filters
#from profilecode import profile
import datetime
import subprocess
import uuid
from scipy.ndimage import morphology as morph
from scipy.ndimage import generic_filter

class ptvio:
    ''' 
    VDC hdf5 file wrapper and data structure handler

    Use this to open vdc files, save vdc objects and manipulate them.
    '''

    def __init__(self,file_name,new=False):
        '''
        Initialise a new vdc instance

        vdcio(file_name,new=False)

        pass True as the second argument to overwrite any existing datasets.
        This deletes them first to overcome some h5py issues with possible
        open files in interactive shells.

        Arguments
           file_name     - string valid .h5 file path
           new           - bool   boolean to delete any existing file

        Returns
           vdc           - vdc object
        '''
        self.file_name = file_name
        self.file_root = os.path.splitext(file_name)[0]
        
        if os.path.isfile(file_name):
            if new==True:
                os.remove(file_name)
            else:
                self._read_header()

    def __str__(self):
        ''' This is returned when you write >> print m # where m is an instance '''
 
        #attributes
        a = ''
        #for key,value in self.viewitems():
        #    a = '%s\n%s:%s' % (a,key.ljust(16), value)
                
        a = 'ix:       %d\n' \
             'iy:       %d\n' \
             'dt:       %d\n' \
             'nt:       %d\n' \
              % (self.ix,self.iy,self.dt,self.nt)
              
        return a 

    def __repr__(self):
        ''' This is returned when you write >> m # where m is an instance '''
        
        return self.__str__()

    def _write_attribute(self,attribute,value,path=''):
        '''
        Write a hdf5 attribute

        _write_attribute(self,attribute,value,path='')

        Arguments
           attribute  - attribute name
           value      - value of attribute
           path       - string OPTIONAL path to group/dataset to add attribute to
        
        
        Returns
           value      - input value
        '''
        with h5py.File(self.file_name,mode='a') as f:
            f.attrs[attribute] = value
        return value

    def _delete_attribute(self,attribute,path=''):
        '''
        Not used
        '''
        # TODO: decide if necessary
        return

    def _read_header(self):
        '''
        Read header information

        _read_header(self)

        Returns
            self
        '''
        with h5py.File(self.file_name, mode='r') as f:
            #if 'data/velocity' in f:
            #    self.frames = {i:f['data/velocity/%s' % i].shape[1] for i in f['data/velocity'].keys()}
           
            if 'comment' in f.attrs:
                self.comment = f.attrs['comment']

            if 'name' in f.attrs:
                self.name = f.attrs['name']

            # assign attributes to method variables.
            [setattr(self,i,f.attrs[i]) for i in list(f.attrs.keys())]

            try:
                self.datasets = {i:f['%s' % i].shape[1] for i in list(f['/'].keys())}
            except:
                error = 1
                
        return self

    def _write_header(self,header):
        '''
        Write header information

        _write_header(self,header)

        Arguments
           header   dictionary of attribute name:value
        
        Returns
           self
        '''
        with h5py.File(self.file_name,mode='a') as f:
            for i in list(header.items()):
                f.attrs[i[0]] = i[1]
        return self._read_header()

    def copy_header(self,ptvio_instance):
        '''
        copy_header(self,vdc_instance)
        
        Copy the header from another vdc_instance

        Argument
           vdc_instance   - vdc instance vdc objec to copy header from

        Return
           self
        '''
        return self._write_header(ptvio_instance._read_header)

    def add_comment(self,comment):
        '''
        add_comment(self,comment)

        Add comment to a file. The comment will be attached to
        a root attribute 'comment' formatted as

        <username> <datetime>       sha:<VERSION>
        <comment>

        This is useful to keep a record of methods/operation tracking of a dataset

        Arguments
           comment    - string comment typically of current action performed

        Returns
           self
        '''

        # if a new file exists this will fail
        # TODO: slightly nicer way?
        try:
            ec = self.comment
        except AttributeError:
            ec = ''

        # ensure it ends with a new line
        if comment[-1:] != '\n':
            comment = '%s\n' % comment

        # current datetime
        dtime = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")

        # build a comment with the user, time and lib version
        comment = '%s%s\t%s\t%s\n %s\n' % (ec,getpass.getuser(),dtime,'sha: #SHA#',comment)

        self.comment = self._write_attribute('comment', comment)
        self._read_header()
        return self

    def set_name(self,file_name):
        '''
        set_name(self,filename)

        Sets the name attribute

        Arguments
           file_name      - string new filename

        Returns:
           self
        '''
        self.name = self._write_attribute('name','%s' % os.path.split(file_name)[-1:])
        
        return self

    def open(self,mode='a'):
        '''
        Open function for file
        '''
        self.f = h5py.File(self.file_name, mode=mode)
        return self

    def close(self):
        self.f.close()
        return self

    def init_group(self,name,description):
        '''
        Initialise a group
        '''
        self.open()
        out = self.f.create_group('%s' % name)
        self.f['%s' %name].attrs['description'] = description
        self.close()
        self._read_header()
        return out

    def get_frames(self):
        with h5py.File(self.file_name, mode='r') as f:
            if 'frames' in f:
                d = list(f['frames/'].keys())
            else:
                d = list(f.keys())
            try:
                d.remove('tracks')
            except:
                pass
            return sorted([int(i[5:]) for i in d])

    def get_length_dist(self):
        with h5py.File(self.file_name, mode='r') as f:
            l =  f['tracks/bylength']
            r = {i.attrs['id']:len(list(i.items())) for i in l.values()}
        
        return r
        
    def get_start_dist(self):
        with h5py.File(self.file_name, mode='r') as f:
            l =  f['tracks/bystart_frame']
            r = {i.attrs['id']:len(list(i.items())) for i in l.values()}
        
        return r
        
    def get_end_dist(self):
        with h5py.File(self.file_name, mode='r') as f:
            l =  f['tracks/byend_frame']
            r = {i.attrs['id']:len(list(i.items())) for i in l.values()}
        
        return r

    def read_frame(self,frame_number):
        ''' 
        Return an array for a single frame
        '''
        with h5py.File(self.file_name, mode='r') as f:
            if 'frames' in f:
                return f['/frames/frame%d' % frame_number][:,:]
            else:
                return f['/frame%d' % frame_number][:,:]
            
    def write_frame(self,frame_number,data):
        ''' 
        Return an array of u components
        '''
        with h5py.File(self.file_name, mode='a') as f:
            f.create_dataset('frames/frame%d' % frame_number, data=data)
    
    def num_tracks(self):
        '''
        Return list of all track keys
        '''
        with h5py.File(self.file_name, mode='r') as f:
            return len(list(f['tracks']['all']))

    def read_track(self,track_number):
        ''' 
        Return an array of u components
        '''
        with h5py.File(self.file_name, mode='r') as f:
            return f['tracks/all/track%d' % track_number][:,:]

    def read_tracks_by_start(self,frame_number):
        ''' 
        Return an array of u components
        '''
        with h5py.File(self.file_name, mode='r') as f:
            s =  f['/tracks/bystart_frame/start_frame%d' % frame_number]            
            r = [i[:] for i in s.values()]
        
        return r
            
    def read_tracks_by_end(self,frame_number):
        ''' 
        Return an array of u components
        '''
        with h5py.File(self.file_name, mode='r') as f:
            e =  f['tracks/byend_frame/end_frame%d' % frame_number]
            r = [i[:] for i in e.values()]
            
        return r
            
    def read_tracks_by_length(self,length):
        ''' 
        Return an array of u components
        '''
        with h5py.File(self.file_name, mode='r') as f:
            l =  f['tracks/bylength/length%d' % length]
            r = [i[:] for i in l.values()]
        
        return r
                
            
    def write_track(self,frame_number,data):
        ''' 
        Return an array of u components
        '''
        with h5py.File(self.file_name, mode='a') as f:
            
            a = f.require_group('tracks/all')
            if 'index' not in a.attrs:
                a.attrs['index'] = 0
            else:
                a.attrs['index'] += 1
                
            t = a.create_dataset('track%d' % a.attrs['index'], data=data)
            t.attrs['id'] = a.attrs['index']
            t.attrs['start'] = frame_number
            t.attrs['end'] = frame_number+data.shape[0]-1
            t.attrs['length'] = data.shape[0]
            
            # create soft links to data based on end and length
            sf = f.require_group('/tracks/bystart_frame/start_frame%d' % frame_number)
            sf.attrs['id'] = frame_number
            f['/tracks/bystart_frame/start_frame%d/track%d'  % (frame_number,a.attrs['index']) ]  = h5py.SoftLink('/tracks/all/track%d' % a.attrs['index'])

            end_frame = frame_number+ data.shape[0]- 1            
            ef = f.require_group('/tracks/byend_frame/end_frame%d' % end_frame)
            ef.attrs['id'] = end_frame
            f['/tracks/byend_frame/end_frame%d/track%d'  % (end_frame,a.attrs['index']) ]  = h5py.SoftLink('/tracks/all/track%d' % a.attrs['index'])            

            l = f.require_group('/tracks/bylength/length%d' % data.shape[0])
            l.attrs['id'] = data.shape[0]
            f['/tracks/bylength/length%d/track%d'  % (data.shape[0],a.attrs['index']) ]  = h5py.SoftLink('/tracks/all/track%d' % a.attrs['index'])            
            
    def status(self,frame_number):
        ''' 
        Return an array of u components
        '''
        with h5py.File(self.file_name, mode='r') as f:
            return f['/frame/frame%d' % frame_number][:,0]
    
    def finalise(self):
        '''
        Rebuild database
        '''

