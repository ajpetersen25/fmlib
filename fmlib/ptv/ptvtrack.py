######################################################################
# DEPENDENCIES
######################################################################
import os
import sys
import gc
from datetime import datetime
import getpass
import inspect
from copy import copy
import argparse
import numpy as np
from scipy import interpolate
from scipy.io import loadmat

from clint.textui import progress
######################################################################
# LOCAL imports
######################################################################
from fmlib.constants import *
from fmlib.ptv.ptvbase import ptvbase
#from profilecode import profile
#from memory_profiler import profile
from fmlib.fio import ptvio
from fmlib.fio import imgio
#from fmlib.io import pivio
from fmlib.tools import *


class ptvtrack(ptvbase):
    ''' 
    Class for cross object.
    Initialise with normal cross commnds. For full instructions call this file
    from command line with -h option
    '''
    
    # fill epsilon value used by fill_vectors
    filleps = 0.1

    def __init__(self,img_filename,    
                      start_frame,
                      track_increment,
                      no_of_pairs,
                      threshold,
                      lowpass,
                      highpass,
                      peak_method,
                      radius,
                      search_radius,
                      Ta=2,
                      dontsave=0,
                      centroid_method=PARTICLE_CENTROID_METHOD,
                      #ref_file=None,
                      particle_position=None,
                      lsq_bounds=PARTICLE_LSQ_BOUNDS
                      ):
        '''
        Initialises instance of cross object coupled to standard algorithm and a single IMG file.
        
        Requires:
           img_filename           - str   - name of IMG file (with extension)
           
           start_frame            - int   - First frame number (numbering starts at 0)
           track_increment        - int   - Increment between images to correlate.
           no_of_frames           - int   - Number of frames to analyse
           threshold              - int   - Threshold
           lowpass                - float - band pass filter gaussian smoothing - 0 is off
           highpass               - int   - band pass filter high pass
           peak_method            - str   - peak method centroid: 'com', localmaxima:'strict', maxmin filtered 'maxfilter', subpixel centroid: 'com_subpixel'
           radius                 - int   - estimated radius size of particles
           search_radius          - int   - search radius for CC around pixels
           Ta                     - int   - search radius threshold for particle acceleration (variation around projected velocity i+1 velocity vector)
           dontsave               - int   - Optional - Don't save tracks less than this length
           centroid_method        - str   - Optional - Defaults to 3-pt gaussian, options are '3pt','lsq'
           ref_file               - str   - Optional - PIV of first pass
           particle_position      - str   - Optional - Default is None, if you already have particle positions add string to .mat file
           lsq_bounds             - int   - Optional - Amount to move the LSQ match.

        See README.md and gitlab wiki for details.
        '''

        # assign the variables
        pair_increment = track_increment
        self._init_vars(start_frame, track_increment, no_of_pairs, pair_increment, threshold,
                              lowpass, highpass, peak_method, radius, search_radius, 0,
                              centroid_method, 1,lsq_bounds)
        
        self.pair_increment = self.track_increment        
        #if self.track_increment != self.pair_increment:
        #    print('track increment %d' % self.track_increment)
        #    print('pair increment %d' % self.pair_increment)
        #    raise Exception('[ERROR] for TR-PTV track_increment (%d) must equal pair_increment (%d)' % (self.track_increment,self.pair_increment))
            
        # init the image instance
        self._init_img(img_filename)

        # read the particle position
        if particle_position is None:
            self.particle_position=None
        else:
            self.read_mat(particle_position)
        # assign track specific variables         
        self._init_vars_track(dontsave,Ta)

        # 4. calculate the first index for each pair
        self._init_frame_index()

        # look for reference PIV
        #if ref_file is None:
        #    self.ref_file = '%s.ref.piv' % (self.file_root)
        #else:
        #    self.ref_file = ref_file
            
        #if os.path.exists(self.ref_file) == 0:
        #    print '[GENERAL] Reference file %s not found.' % self.ref_file
        self.initial_gd = self.noref_gd
        #    self.ref_file_exists = False
        #    self.ref_file = ''
        #else:
        #    print '[GENERAL] Reference PIV file %s found.' % self.ref_file
       #     self.ref_piv = pivio.pivio(self.ref_file)
       #     self.px = self.ref_piv.x
       #     self.py = self.ref_piv.y
        #    self.initial_gd = self.ref_gd
        #    self.ref_file_exists = True
            
        # 7. init ptv
        self._init_ptv()
    
    def read_mat(self, particle_position):
        # load .mat file, C[:,1]=x, C[:,2]=y particle positions
        mat = loadmat(particle_position)
        C = mat['centers']     
        self.particle_position = np.squeeze(C)    
        
    def __str__(self):
        ''' 
        This is returned when you write >> print m # where m is an instance
        '''
        s = 'ptv instance for generating PTV output from IMG data\n\n' \
            'img file: %s\n'  \
            '      ix: %d\n'  \
            '      iy: %d\n'  \
            '      it: %d\n'  \
            '\n' \
            'cross arguments:\n' \
            '      start frame:     %d\n' \
            '      track increment: %d\n' \
            '      no of pairs:     %d\n' \
            '      lowpass          %.2f\n' \
            '      highpass         %d\n' \
            '      peak_method      %s\n' \
            '      radius:          %d\n' \
            '      search radius:   %d\n' \
            '      T_a:   %d\n' \
            '      dont save unless track is at least %d long\n' \
            '      centroid method  %s\n' \
            '      lsq:             %d\n' % (self.img.file_name, 
                                             self.img.ix, self.img.iy, self.img.it,
                                             self.start_frame, 
                                             self.track_increment, 
                                             self.no_of_pairs,
                                             self.lowpass,
                                             self.highpass,
                                             self.peak_method,
                                             self.radius,
                                             self.search_radius,
                                             self.Ta,
                                             self.dontsave,
                                             self.centroid_method,
                                             #self.ref_file,
                                             self.lsq_bounds )
        
        return s

    def __repr__(self):
         ''' 
         This is returned when you write >> m # where m is an instance 
         '''
         return "cross instances at "+hex(id(self))


    def _init_vars_track(self, dontsave,Ta):
        '''
        Initialise instance variables for standard cross-correlation and assigns to instance variables

        Arguments:
          same as class

        Returns:
          None
        '''
        
        self.dontsave = dontsave
        self.passes = 1 
        self.maxtracks=MAXTRACKS
        self.tracks = [None]*self.maxtracks
        self.nooftracks = 0
        self.Ta = Ta
        
    def _init_ptv(self):
        '''
        Initialises the ptv objects and
        assigns them to instance variables

        Arguments:
          None

        Returns:
          None
        '''
        self.ptv = ptvio.ptvio('%s.track.ptv' % (self.file_root),new=True)
        self.ptv.add_comment(self.gen_ptv_comment())
        self.ptv._write_attribute('ix',self.img.ix)
        self.ptv._write_attribute('iy',self.img.iy)
        self.ptv._write_attribute('nt',self.no_of_pairs)
        self.ptv._write_attribute('dt',self.track_increment)

    def gen_ptv_comment(self):
        '''
        Generate the ptv comments
        '''
        d = datetime.now()

        return "%s\n%s %s %d %d %d %d %.1f %d %s %d %d %d %d %s %s %d\nfmlib git sha: #SHA#\n%s\n\n" % (getpass.getuser(), 
                                                                                 inspect.stack()[-1][1],
                                                                                 self.img.file_name,
                                                                                 self.start_frame,
                                                                                 self.track_increment,
                                                                                 self.no_of_pairs,
                                                                                 self.threshold,
                                                                                 self.lowpass,
                                                                                 self.highpass,
                                                                                 self.peak_method,
                                                                                 self.radius,
                                                                                 self.search_radius,
                                                                                 self.Ta,
                                                                                 self.dontsave,
                                                                                 self.centroid_method,
                                                                                 #self.ref_file,
                                                                                 self.particle_position,
                                                                                 self.lsq_bounds,
                                                                                 datetime.now().strftime("%a %b %d %H:%M:%S %Y"))
    
                                                                                 
    def noref_gd(self,i):
        '''
        Returns zero initial offsets for the x,y displacements
        '''
        return np.zeros(self.xgl.shape),np.zeros(self.ygl.shape)
        
    """def ref_gd(self,frame_no):
        '''
        Returns initial offsets for the x,y displacements based on PIV interpolation
        
        Arguments:
          frame_no   - frame number to use. Ignored if ref_piv has 1 frame (average)
          
        Returns:
          u,v        - tuple of horizontal and vertical velocities at xgl,ygl locations
        '''
        if self.ref_piv.nt == 1:
            get_frame = 0
        else:
            get_frame = (frame_no-self.ref_piv.t0)/self.ref_piv.rt
        
        self.fu = interpolate.RectBivariateSpline(self.py,self.px, self.ref_piv.u(get_frame))
        self.fv = interpolate.RectBivariateSpline(self.py,self.px, self.ref_piv.v(get_frame))
        return self.fu.ev(self.ygl, self.xgl)/self.track_increment*self.ref_piv.dt,self.fv.ev(self.ygl, self.xgl)/self.track_increment*self.ref_piv.dt"""
    
    def add_tracks(self,tracks):
        for track in tracks:
            self.add_track(track)
            
    def add_track(self,track):
        if self.nooftracks==self.maxtracks:
            self.maxtracks*=4
            nm = [None]*self.maxtracks
            nm[:self.nooftracks] = self.tracks
            self.tracks = nm
        
        self.tracks[self.nooftracks] = track
        self.nooftracks += 1
        
    def del_tracks(self,tracks):
        for track in tracks:
            self.del_track(track)

    #def del_track(self,track):
    #    map(self.tracks.__getitem__, self.active)
    def prune_tracks(self,tracks):
        pruned_tracks = list(map(self.tracks.__getitem__, tracks))
        self.nooftracks = len(pruned_tracks)
        self.tracks[0:self.nooftracks] = pruned_tracks

    def tracking(self):
        if self.particle_position is None:
            self.track_all_frames()
        else:
            self.track_all_frames_withpos()
            
    def track_all_frames(self):
        '''
        Crosses all frames specified by instance initialisation.
        This version assumes a 'ref' PIV file use based on the same values
        as the x_last_pass size as the input guess.

        1. writes the PIV headers
        2. Loops through frames to process
        3. Call cross and writes results to PIV

        Arguments:
          None

        Returns:
          None
        '''

        # write PIV headers
        #self.ptv.write_header()

        # process each frames
        k = 0

        i = self.frames_to_process[0]
        k = k+1
        print('Image pair number : %03d ( %03d - %03d )' % (k,i,i+self.track_increment))
        print()

        # Read in data
        frameA = self.img.read_frame2d(i)
        frameB = self.img.read_frame2d(i+self.track_increment)

        if self.lowpass > 0:
            frameA = bandpass(frameA,self.lowpass,self.highpass,threshold=self.threshold)
            frameB = bandpass(frameB,self.lowpass,self.highpass,threshold=self.threshold)
            threshold = self.threshold
        else:
            threshold = self.threshold

        # get the peaks.
        pframeA = peak_finder_func(self.peak_method,frameA,self.radius,threshold=threshold,crop=self.crop)
        pframeB = peak_finder_func(self.peak_method,frameB,self.radius,threshold=threshold,crop=self.crop)

        if self.peak_method != 'com_subpixel':
            pframeA = registration_array(self.centroid_method,frameA,pframeA,self.radius,self.lsq_bounds)
            pframeB = registration_array(self.centroid_method,frameB,pframeB,self.radius,self.lsq_bounds)

        if pframeA.shape[0]>MAXPARTICLES:
            sys.exit('Too many particles %d. Hard limit is %d for memory allocation. TODO' % (pframeA.shape[0],MAXPARTICLES))

        self.xgl = pframeA[:,0]
        self.ygl = pframeA[:,1]

        self.xgl2 = pframeB[:,0]
        self.ygl2 = pframeB[:,1]

        # self.ref is definted to get_norefgd or get_refgd depending on PIV ref file existance
        self.xgd,self.ygd = self.initial_gd(i)

        # TODO nearest neighbour replace with RM
        self.add_tracks([track(self.start_frame,self.xgl[s],self.ygl[s],s) for s in range(len(self.xgl))])

        self.active = np.arange(self.nooftracks)

        self.track_frames1(i,i+self.track_increment)

        #TODOself.print_status(self.status)

        for i in self.frames_to_process[2:]:
            k = k+1
            print()
            print('Image pair number : %03d ( %03d - %03d )' % (k,i-self.track_increment,i))
            print()

            # Read in data
            frameA = self.img.read_frame2d(i)
            frameB = self.img.read_frame2d(i+self.track_increment)

            if self.lowpass > 0:
                frameA = bandpass(frameA,self.lowpass,self.highpass,threshold=self.threshold)
                frameB = bandpass(frameB,self.lowpass,self.highpass,threshold=self.threshold)
                threshold = self.threshold
            else:
                threshold = self.threshold

            pframeA = peak_finder_func(self.peak_method,frameA,self.radius,threshold=threshold,crop=self.crop)
            pframeB = peak_finder_func(self.peak_method,frameB,self.radius,threshold=threshold,crop=self.crop)

            if self.peak_method != 'com_subpixel':
                pframeA = registration_array(self.centroid_method,frameA,pframeA,self.radius,self.lsq_bounds)
                pframeB = registration_array(self.centroid_method,frameB,pframeB,self.radius,self.lsq_bounds)

            if pframeA.shape[0]>MAXPARTICLES:
                sys.exit('Too many particles %d. Hard limit is %d for memory allocation. TODO' % (pframeA.shape[0],MAXPARTICLES))

            self.xgl = pframeA[:,0]
            self.ygl = pframeA[:,1]

            self.xgl2 = pframeB[:,0]
            self.ygl2 = pframeB[:,1]

            self.xgd,self.ygd = self.initial_gd(i)

            self.track_frames2(i,i+self.track_increment)

        # close all tracks
        extra_track = 0
        for t in self.tracks:
            if t is not False and t is not None:
                if (t.finished == 0) & ((t.current-1) > self.dontsave):
                    self.ptv.write_track( t.start, np.column_stack( (t.x, t.y, list(range(t.start,t.stop+1)) ) ) )
                    extra_track +=1

        print('%d tracks saved on final frame' % extra_track)
        return

    def track_all_frames_withpos(self):
        '''
        Crosses all frames specified by instance initialisation.
        This version assumes a 'ref' PIV file use based on the same values
        as the x_last_pass size as the input guess.

        1. writes the PIV headers
        2. Loops through frames to process
        3. Call cross and writes results to PIV

        Arguments:
          None

        Returns:
          None
        '''

        # write PIV headers
        #self.ptv.write_header()

        # process each frames
        k = 0

        i = self.frames_to_process[0]
        k = k+1
        print('Image pair number : %03d ( %03d - %03d )' % (k,i,i+self.track_increment))
        print

        #read particle positions for the first two frames
        pframeA = self.particle_position[i]
        pframeB = self.particle_position[i+self.track_increment]

        self.xgl = pframeA[:,0]
        self.ygl = pframeA[:,1]
        
        self.xgl2 = pframeB[:,0]
        self.ygl2 = pframeB[:,1]
        
        # self.ref is definted to get_norefgd or get_refgd depending on PIV ref file existance
        self.xgd,self.ygd = self.initial_gd(i)         
        
        # TODO nearest neighbour replace with RM
        self.add_tracks([track(self.start_frame,self.xgl[s],self.ygl[s],s) for s in range(len(self.xgl))])
        
        self.active = np.arange(self.nooftracks)
        
        self.track_frames1(i,i+self.track_increment)
        
        #TODOself.print_status(self.status)
        
        for i in self.frames_to_process[2:]:
            k = k+1
            print() 
            print('Image pair number : %03d ( %03d - %03d )' % (k,i-self.track_increment,i))
            print()

            pframeA = self.particle_position[i]
            pframeB = self.particle_position[i+self.track_increment]

            self.xgl = pframeA[:,0]
            self.ygl = pframeA[:,1]
        
            self.xgl2 = pframeB[:,0]
            self.ygl2 = pframeB[:,1]
            
            self.xgd,self.ygd = self.initial_gd(i)
            
            self.track_frames2(i,i+self.track_increment)
            
        # close all tracks
        extra_track = 0
        for t in self.tracks:
            if t is not False and t is not None:
                if (t.finished == 0) & ((t.current-1) > self.dontsave):
                    self.ptv.write_track( t.start, np.column_stack( (t.x, t.y, range(t.start,t.stop+1) ) ) )
                    extra_track +=1
                    
        print('%d tracks saved on final frame' % extra_track)
        return
 
    def track_frames1(self,frameA_index,frameB_index,debug=None):
        '''
        Performs cross correlation of two frames, subpixel registration, reject, fill routines
        Depending on the p value it performs a discrete pixel offset at the beginning

        Arguments:
           frameA_index - int of img index for first frame
           frameB_index - int of img index for second frame
           debug        - debug

        '''

        # only when no information is available.
        used = []
        xy2x = (np.repeat(self.xgl+self.xgd,len(self.xgl2)) - np.tile(self.xgl2,len(self.xgl))).reshape(len(self.xgl),len(self.xgl2))
        xy2y = (np.repeat(self.ygl+self.ygd,len(self.xgl2)) - np.tile(self.ygl2,len(self.xgl))).reshape(len(self.xgl),len(self.xgl2))
        xy2d = xy2x**2+xy2y**2
        lost = 0
        
        # Loop through all vector locations
        for k in range(len(self.xgl)):
            if np.min(xy2d) < self.srs:
                xi,x2i = np.unravel_index(np.argmin(xy2d),xy2d.shape)
                if x2i not in used:
                    self.tracks[xi].add_frame(self.xgl2[x2i],self.ygl2[x2i],1)
                    used.append(x2i)
                    xy2d[:,x2i] = np.inf
                    xy2d[xi,:] = np.inf
                else:
                    self.tracks[xi].end()
                    self.active = self.active[self.active != xi]
            else:
                lost += 1
        
        # find any outliers and remove them from active
        xi,x2i = np.where(xy2d < np.inf)
        for i in np.unique(xi):
            self.tracks[i].end()
            self.active = self.active[self.active != i]
        
        new = [track(self.start_frame+self.track_increment,self.xgl2[i],self.ygl2[i],1) for i in (np.setdiff1d(range(len(self.xgl2)), used))];
        #array of track indices which only have one 
        print('Frame %05d' % self.start_frame)
        print('%05d particles found' % len(self.xgl))
        print('%05d tracks made' % len(self.xgl))
        print()
        print('Frame %05d' % (self.start_frame+self.pair_increment))
        print('%05d particles found' % len(self.xgl2))
        print('%05d track continued' % len(used))
        print('%05d tracks ended' % lost)
        print('%05d new points' % len(new))
        
        #save used into a frame
        sxyuv = np.zeros((len(self.active),5))
        for idx,i in enumerate(self.active):
            sxyuv[idx,:] = [ 1, self.tracks[i].x[-2], self.tracks[i].y[-2], self.tracks[i].u[-1], self.tracks[i].v[-1] ]
            
        self.ptv.write_frame( self.start_frame, sxyuv)
        self.prune_tracks(self.active)
        self.add_tracks(new)
        self.active = np.arange(self.nooftracks)
        
    #@profile
    def track_frames2(self,frameA_index,frameB_index,debug=None):
        '''
        Performs cross correlation of two frames, subpixel registration, reject, fill routines
        Depending on the p value it performs a discrete pixel offset at the beginning

        Arguments:
           frameA_index - int of img index for first frame
           frameB_index - int of img index for second frame
           debug        - debug

        '''
        used = []
        finished = []
        active2 = copy(self.active)
        nofriends = 0
        otherfriends = 0

#TODO        # only when no information is available.
#        xy2x = (np.repeat(self.xgl+self.xgd,len(self.xgl2)) - np.tile(self.xgl2,len(self.xgl))).reshape(len(self.xgl),len(self.xgl2))
#        xy2y = (np.repeat(self.ygl+self.ygd,len(self.xgl2)) - np.tile(self.xgl2,len(self.xgl))).reshape(len(self.xgl),len(self.xgl2))
#        xy2d = xy2x**2+xy2y**2

        lost = 0
        
        # Loop through all current active paths
        for j in self.active:
            
            if self.tracks[j].current > 1:
                u = self.tracks[j].u[-1]
                v = self.tracks[j].v[-1]
                srs = self.Ta**2
            #elif self.ref_file_exists:
            #    u = self.fu.ev(self.tracks[j].lasty,self.tracks[j].lastx)*self.pair_increment/self.ref_piv.dt
            #    v = self.fv.ev(self.tracks[j].lasty,self.tracks[j].lastx)*self.pair_increment/self.ref_piv.dt
            #    srs = self.Ta**2
            else:
                u = 0
                v = 0
                srs = self.srs
                
            # xj(n+1) matches in search radius for 1st order (minimum accel)
            d = np.where(((self.tracks[j].lastx +u-self.xgl)**2+ (self.tracks[j].lasty+v-self.ygl)**2)<srs)[0]
            t = np.zeros((len(d),2))
            t[:,0] = self.xgl[d]
            t[:,1] = self.ygl[d]
            
            #if (i == 2) & debug == 1:
            # if possible matches exist in both n+1 and n+2
            
            if len(d):
                # build cost function with max friend assumption
                cf = np.empty((len(d),MAXFRIENDS))
                cf.fill(np.inf)
                
                # loop through all n+1
                for ik,k in enumerate(t):
                    u1 = k[0] - self.tracks[j].lastx
                    v1 = k[1] - self.tracks[j].lasty
                    # get the (n+2) matches TODO - ony get n+2 base on n+1 (t2 nested in t?)
                    d2 = np.where(((self.tracks[j].lastx +u+u1-self.xgl2)**2+ (self.tracks[j].lasty+v+v1-self.ygl2)**2)<srs)[0]
                    #d2 = np.where(((self.tracks[j].lastx +u1+u2-self.xgl2)**2+ (self.tracks[j].lasty+v1+v2-self.ygl2)**2)<self.srs)[0]
                    if len(d2):
                        xx = self.xgl2[d2] - (self.tracks[j].lastx +u+u1  )
                        yy = self.ygl2[d2] - (self.tracks[j].lasty +v+v1  )
                        cf[ik,0:len(xx)] = xx**2+yy**2
    
                # find minimal value cost function n+1 value
                ii,jj = np.unravel_index(np.argmin(cf), cf.shape)
                
                # d[ii] is the value of the n+1 in t
                if d[ii] not in used:
                    # KEEP THEM
                    try:
                        self.tracks[j].add_frame(t[ii,0],t[ii,1],1)
                    except:
                        print(i)
                        
                    used.append(d[ii])
                else:
                    # DROP THEM - OTHER FRIENDS
                    active2 = active2[active2 != j]   
                    otherfriends += 1
                    self.tracks[j].end()
                    finished.append(j)
    
            else:
                # DROP THEM - NO FRIENDS
                nofriends += 1
                self.tracks[j].end()
                finished.append(j)
                active2 = active2[active2 != j]
                
        new = [track(frameA_index,self.xgl[z],self.ygl[z],1) for z in (np.setdiff1d(range(len(self.xgl)), used))]
        
        saved = 0
        dropped = 0
        for i in finished:
            if (self.tracks[i].current-1) > self.dontsave:
                self.ptv.write_track( self.tracks[i].start, np.column_stack( (self.tracks[i].x,self.tracks[i].y, range(self.tracks[i].start,self.tracks[i].stop+1) ) ) )
                saved += 1
            else:
                dropped += 1
        
        print('Frame %05d' % frameA_index)
        print('%05d particles found' % len(self.xgl))
        print('%05d previous tracks' % len(self.active))
        print('%05d track continued' % len(used))
        print('%05d tracks ended' % nofriends)
        print('%05d tracks conflicted' % otherfriends)
        print('%05d new tracks' % len(new))
        print('%05d future tracks' % (len(used)+len(new)))
        print('%05d tracks saved' % saved)
        if len(self.active) > 0:
            print('%d%% continued' % ((len(self.active)-nofriends-otherfriends)/len(self.active)*100 ))
        #save used into a frame
        sxyuv = np.zeros((len(active2),5))
        for idx,i in enumerate(active2):
            sxyuv[idx,:] = [ 1, self.tracks[i].x[-2], self.tracks[i].y[-2], self.tracks[i].u[-1], self.tracks[i].v[-1] ]
        
        self.ptv.write_frame( (frameA_index-self.track_increment),sxyuv)
        
        self.prune_tracks(active2)
        self.add_tracks(new)
        self.active = np.arange(self.nooftracks)
        gc.collect()

def track_parse():
    '''
       Parse the inputs from commandline
    '''
    def peakstr(string):
        if string.lower() not in ['com','strict','maxfilter', 'com_subpixel']:
            raise argparse.ArgumentTypeError('"%s" is not a valid peak detection method' % string)
        return string.lower()
        
    def centroidstr(string):
        if string.lower() not in ['3pt','polynomial','lsq']:
            raise argparse.ArgumentTypeError('"%s" is not a valid centroid detection method' % string)
        return string.lower()
        
    parser = argparse.ArgumentParser(description='PTV 4BE code', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('img_file', type=str,
                        help='IMG filename for cross-correlation')
    parser.add_argument('start_frame', type=int,
                        help='First frame number (numbering starts at 0)')
    parser.add_argument('track_increment', type=int,
                        help='Increment between images to track - 0 only extracts the locations')
    parser.add_argument('no_of_pairs', type=int,
                        help='Number of frames to analyse')
    parser.add_argument('threshold', type=int,
                        help='Threshold the data (after bandpass if it is chosen)')    
    parser.add_argument('lowpass', type=float,
                        help='Low pass filter gaussian smoothing - 0 will skip highpass as well')
    parser.add_argument('highpass', type=int,
                        help='High pass filter - 0 turns it off')
    parser.add_argument('peak_method', type=peakstr,
                        help='Peak detection, one of com, strict or maxfilter')
    parser.add_argument('radius', type=int,
                        help='Expected radius of object')
    parser.add_argument('search_radius', type=int,
                        help='Search radius - if .piv is give, it WILL be the estimate of max RMS in pixels on mean IW flow')
    parser.add_argument('Ta', type=int,
                        help='search radius threshold for particle acceleration (variation around projected velocity i+1 velocity vector)')
    parser.add_argument('dontsave', type=int, nargs='?', default=0,
                        help='Optional - Minimum number of track lengths to save')
    parser.add_argument('centroid_method', type=centroidstr, nargs='?',default=PARTICLE_CENTROID_METHOD, 
                        help='Optional - centroid-finding method, default: %s' % PARTICLE_CENTROID_METHOD ) 
    #parser.add_argument('ref_file', type=str, nargs='?', default=None,
    #                    help='Optional - Reference PIV file for first pass estimate')
    parser.add_argument('particle_position', type=str, nargs='?',default=None,
                        help='Particle positions')
    parser.add_argument('lsq_bounds',type=int,nargs='?',default=PARTICLE_LSQ_BOUNDS,
                        help='Optional, if setting bounds, can either be scalars which apply to all parameters, or arrays with the same length as parameters. e.g. lower bound=(intensity, centroid_row, centroid_col, particle diameter)')

    args = parser.parse_args()
    """print()
    print('img file: %s, start frame: %d, track increment: %d, no_of_pairs: %d, threshold: %d, lowpass: %.2f \n \
            highpass: %d, peak method: %s, radius: %d, search radius: %d, Ta: %d, dontsave: %d, centroid method: %s \n \
            particle position: %s, lsq bounds: %d \n' %(args.img_file,args.start_frame,args.track_increment,args.no_of_pairs,args.threshold,args.lowpass,args.highpass,args.peak_method,args.radius,args.search_radius,args.Ta,args.dontsave,args.centroid_method,args.particle_position,args.lsq_bounds))
    print()"""
    fail = False
    # check if IMG file exists
    if os.path.exists(args.img_file) == 0:
        print('[ERROR] IMG file does not exist')
        fail = True

    img_root, img_ext = os.path.splitext(args.img_file)
    
    # check if particle position file exists
    if args.particle_position is not None:
        if os.path.exists(args.particle_position) == 0:
            print('[ERROR] particle position file does not exist')
            fail = True

    # check if REF PIV file exists for deformation
    if fail:
        os.sys.exit('ERRORS FOUND. Exiting...')
    return args

class track():
    '''
    track object for keeping a record of particle tracks
    '''
    def __init__(self,start_frame,x,y,s):
        self.start = start_frame
        self.stop = start_frame
        self._x = np.zeros((100,))
        self._y = np.zeros((100,))
        self._s = np.zeros((100,))
        self.current = 1
        self.max = 100
        self._x[0] = x
        self._y[0] = y
        self.finished = 0
        
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        '''
        Short description of instance
        '''
        return 'Track: start %d, stop %d, length %d, finished %d, current xy %.2f,%.2f' % (self.start,self.stop, self.current, self.finished, self.lastx, self.lasty)
        
    def add_frame(self, x,y,s):
        
        if self.current == self.max:
            self.max *= 4
            newdata = np.zeros((self.max,))
            newdata[:self.current] = self._x
            self._x= newdata
            
            newdata = np.zeros((self.max,))
            newdata[:self.current] = self._y
            self._y = newdata

        self._x[self.current] = x
        self._y[self.current] = y
        self.current += 1
        self.stop += 1

    def end(self):
        self.finished = 1
        self._x = np.array(self._x[:self.current])
        self._y = np.array(self._y[:self.current])
    
    @property
    def x(self):
        return self._x[:self.current]
    
    @property
    def y(self):
        return self._y[:self.current]
        
    @property
    def u(self):
        return np.diff(self._x[:self.current])
    
    @property
    def v(self):
        return np.diff(self._y[:self.current])
        
    @property
    def lastx(self):
        return self._x[self.current-1]
    
    @property
    def lasty(self):
        return self._y[self.current-1]

