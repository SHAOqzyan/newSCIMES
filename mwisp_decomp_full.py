import os
import sys

import numpy as np

from astropy import wcs
from astropy.io import fits
from scipy.ndimage.measurements import label

from astropy.table.table import Column
from astropy.table import Table
from astrodendro import Dendrogram, ppv_catalog
from scimes import SpectralCloudstering

from matplotlib import pyplot as plt
import matplotlib.colors as colors
import time

import glob

from pdb import set_trace as stop

from spectral_cube import SpectralCube


def mad(x):

	x = x[np.isfinite(x)]
	y = np.nanmedian(np.abs(x-np.nanmedian(x)))/0.6745

	return y



def dilmasking(hdu, fchan = 8, ed = 5, th = 3,rms=0.5):

    data = hdu.data
    hd = hdu.header
    
    #free1 = data[0:fchan,:,:]
    #rms = np.nanstd(free1[np.isfinite(free1)])

    rmsmap = data[0]*0+rms  #np.nanstd(free1, axis=0)
    rmscube = np.zeros(data.shape)
    s2ncube = np.zeros(data.shape)

    for v in range(data.shape[0]):
        s2ncube[v,:,:] = data[v,:,:]/rmsmap[:,:]
        rmscube[v,:,:] = rmsmap[:,:]

    mask1 = np.zeros(data.shape, dtype=np.int)
    mask2 = np.zeros(data.shape, dtype=np.int)

    mask1[s2ncube > ed] = 1
    mask2[s2ncube > th] = 1

    mask1o = np.zeros(data.shape)
    mask2o = np.zeros(data.shape)
    # mask1 is large than 5 sigma?

    # continuously, there must be three channels both large than 5 sigma,#
    mask1o[mask1*(np.roll(mask1,1,axis=0)+np.roll(mask1,-1,axis=0)) > 0] = 1
    mask2o[mask2*(np.roll(mask2,1,axis=0)+np.roll(mask2,-1,axis=0)) > 0] = 1

    maskt = mask1o+mask2o

    # Connectivity structure
    s = np.asarray([[[1,1,1],[1,1,1],[1,1,1]],\
                    [[1,1,1],[1,1,1],[1,1,1]],\
                    [[1,1,1],[1,1,1],[1,1,1]]])

    # Region labelling
    labarr, nfeats = label(mask2o, structure=s)

    #print nfeats, " islands found"


    # Eliminate the clouds without core:
    # applying the mask 1 the islands with
    # core only stay -> find their asgns
    flabarr = labarr*mask1o
    asgns = np.unique(flabarr)
    asgns = asgns.astype(np.int)
    asgns = asgns[asgns != 0]

    # Crate the dilate mask with only
    # the selected islands
    dilmask = np.zeros(data.shape, dtype=np.int)   

    vs, ys, xs  = np.where(labarr > 0)
    ids = labarr[np.where(labarr > 0)]
    xys = xs+ys*data.shape[0]

    sids = np.in1d(ids,asgns)
    sxs = xs[sids]
    sys = ys[sids]
    svs = vs[sids]

    dilmask[(svs,sys,sxs)] = 1
    data[dilmask == 0] = np.nan

    dilmask = fits.PrimaryHDU(dilmask,hd)
    data = fits.PrimaryHDU(data,hd)
    labarr = fits.PrimaryHDU(labarr,hd)

    return dilmask, data, labarr




def listfiles(fname,path):
  
  pfilenames = glob.glob(path+fname)
  filenames = []
  
  for pfilename in pfilenames:
    
    filenames.append(pfilename[len(path)::])
    
  return filenames




run = 2

mpath = './' # Main working directory
path = mpath+'SCIMES/'+str(run)+'_RUN/' # SCIMES directory with the two run folders
dpath = mpath+'SCIMES/DATA/'
upath = mpath+'SCIMES/UDATA/'

do_r1subfiles = False # To create the names of the run 1 subfiles
do_r2subfiles = False # To create the names of the run 2 subfiles
do_slicing = True # Divide the survey full cubes into subcubes
do_scimes = True # Run dendrogram and SCIMES

# These ones might not be necessary
do_edgecloud = False
do_edgecloudcoors = False


# Increase the recursion limit
sys.setrecursionlimit(100000)


# Some parameters / names to set globally

bmaj = 50./3600 # Modify with the size of the beam in deg
bmin = 50./3600 # Modify with the size of the beam in deg

#full_cube_name = 'MWISP' # Full cube name
full_cube_name = 'G214CO12' # Full cube name


#%&%&%&%&%&%&%&%&%&%&%&%&%&%        
#   Sub-files for Run 1
#%&%&%&%&%&%&%&%&%&%&%&%&%&%

if do_r1subfiles:
    #
    print   os.path.isdir(path+'SUBCUBES/')
    if os.path.isdir(path+'SUBCUBES/') == False:
        os.system('mkdir '+path+'SUBCUBES/') 

    fcube_hd = fits.getheader(mpath+full_cube_name+'.fits')

    subcubes = [] 

    cdelt = np.round(np.abs(fcube_hd['CDELT1'])*3600.)
    sx = np.int(2*3600/cdelt) # Wanted longitudinal size of the subcube, here 2 deg
    csize = fcube_hd['NAXIS1']

    i = 0
    j = 0

    print("Slicing...") 

    while i + sx < csize:

        xi = i
        xf = i + sx

        subcubes.append('mwisp'+str(j).zfill(2)+'_'+str(xi)+'_'+str(xf))

        i += sx
        j += 1
      
    xi = i
    xf = csize

    subcubes.append('mwisp_'+str(j).zfill(2)+'_'+str(xi)+'_'+str(xf))



    np.save(path+'SUBCUBES/subfiles.npy', subcubes)


#%&%&%&%&%&%&%&%&%&%&%&%&%&%        
#   Sub-files for Run 2
#%&%&%&%&%&%&%&%&%&%&%&%&%&%

if do_r2subfiles:

    if os.path.isdir(path+'SUBCUBES/') == False:
        os.system('mkdir '+path+'SUBCUBES/') 

    fcube_hd = fits.getheader(mpath+full_cube_name+'.fits')

    subcubes = [] 

    cdelt = np.round(np.abs(fcube_hd['CDELT1'])*3600.)
    sx = np.int(2*3600/cdelt)
    csize = fcube_hd['NAXIS1'] - sx/2

    i = sx/2
    j = len(np.load(mpath+'SCIMES/1_RUN/SUBCUBES/subfiles.npy'))

    print("Slicing...") 

    while i + sx < csize:

        xi = i
        xf = i + sx

        subcubes.append('mwisp_'+str(j).zfill(2)+'_'+str(xi)+'_'+str(xf))

        i += sx
        j += 1
      
    xi = i
    xf = csize

    subcubes.append('mwisp_'+str(j).zfill(2)+'_'+str(xi)+'_'+str(xf))

    np.save(path+'SUBCUBES/subfiles.npy', subcubes)




#%&%&%&%&%&%&%&%&%&%&%&%&%&%        
#       Slicing
#%&%&%&%&%&%&%&%&%&%&%&%&%&%

if do_slicing:

    fcube = SpectralCube.read(mpath+full_cube_name+'.fits')

    subfiles = np.load(path+'SUBCUBES/subfiles.npy')



    print("Slicing...") 
  
    #for j,subfile in enumerate(subfiles):
    for j in range(len(subfiles)):

        subfile = subfiles[j]
        print(subfile)

        xi = int(subfile.split('_')[2])
        xf = int(subfile.split('_')[3])
        scube = fcube[:,:,xi:xf]

        print(subfile, scube.hdu.data.shape, xf-xi)
        scube.write(path+'SUBCUBES/'+subfile+'.fits',format='fits',overwrite=True)

    fcube = None



#%&%&%&%&%&%&%&%&%&%&%&%&%&%        
#     SCIMES it!!
#%&%&%&%&%&%&%&%&%&%&%&%&%&%

from astropy import units as u
if do_scimes:

    #plt.switch_backend('pdf')

    # Check directory existence
    if os.path.isdir(path+'CLUST_AFFS/') == False:
        os.system('mkdir '+path+'CLUST_AFFS/') 

    if os.path.isdir(path+'DENDROGRAMS/') == False:
        os.system('mkdir '+path+'DENDROGRAMS/')

    if os.path.isdir(path+'CATALOGS/') == False:
        os.system('mkdir '+path+'CATALOGS/')    

    if os.path.isdir(path+'ASGNCUBES/') == False:
        os.system('mkdir '+path+'ASGNCUBES/')    

    if os.path.isdir(path+'LEAFASGNCUBES/') == False:
        os.system('mkdir '+path+'LEAFASGNCUBES/')    

    if os.path.isdir(path+'TRUNKASGNCUBES/') == False:
        os.system('mkdir '+path+'TRUNKASGNCUBES/')    

    if os.path.isdir(path+'TREES/') == False:
        os.system('mkdir '+path+'TREES/')  
        
            
    files = np.load(path+'SUBCUBES/subfiles.npy')
    
    runfiles = range(len(files))

    t = time.time()

    setup = '_saveall'

    for f in runfiles:#[runfile]:

        print("%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&&%&")
        print("---Performing", files[f], "File:", f+1, "/", len(files))
        print("%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&&%&")

        print("Loading the files...")
        hdu = fits.open(path+'SUBCUBES/'+files[f]+'.fits')[0]
        hd = hdu.header
        data = hdu.data
        sizes = data.shape

        # Take the first and last 10 line-free channels to calculate the RMS map
        free1 = data[sizes[0]-11:sizes[0]-1,:,:]
        free2 = data[0:10,:,:]
        free = np.concatenate([free2,free1],axis=0)
        rmsmap = np.nanstd(free, axis = 0)
        rms = np.nanmedian(rmsmap)


        # Mask the data with the dilate mask technique. See Rosolowsky & Leroy 2006.
        _, mask, _ = dilmasking(hdu, ed = 5, th = 3, rms=0.5  )
        data[mask.data == 0] = np.nan

        mask2 = None
        hdu = None
        free, free1, free2 = None, None, None
        rmsmap = None


        cdelt1 = abs(hd.get('CDELT1'))
        cdelt2 = abs(hd.get('CDELT2'))
        ppbeam = abs((bmaj*bmin)/(cdelt1*cdelt2)*2*np.pi/(8*np.log(2))) # Pixels per beam


        if os.path.isfile(path+'DENDROGRAMS/'+files[f]+setup+'_dendrogram.fits') == False:
            print("Making dendrogram...")

            from pruning import all_true, min_vchan, min_delta, min_area
            is_independent = all_true((min_delta(3*rms), min_area(3*ppbeam), min_vchan(2)))

            d = Dendrogram.compute(data, min_value=0, is_independent=is_independent, verbose=1)

            d.save_to(path+'DENDROGRAMS/'+files[f]+setup+'_dendrogram.fits')

        else:
            print("Loading dendrogram...")
            d = Dendrogram.load_from(path+'DENDROGRAMS/'+files[f]+setup+'_dendrogram.fits')


        if os.path.isfile(path+'CATALOGS/'+files[f]+setup+'_catalog.fits') == False:
            print("Making a simple catalog of dendrogram structures...")
            metadata = {}
            metadata['data_unit'] = u.Jy #This must be Kelvin, but astrodendro gets only Jy, so fake Jy as unit

            cat = ppv_catalog(d, metadata)
            cat['luminosity'] = cat['flux'].data
            cat['volume'] = np.pi*cat['radius'].data**2*cat['v_rms'].data 

            os.system('rm -rf '+path+'CATALOGS/'+files[f]+setup+'_catalog.fits')
            cat.write(path+'CATALOGS/'+files[f]+setup+'_catalog.fits')
        else:
            print("Loading catalog...")
            cat = Table.read(path+'CATALOGS/'+files[f]+setup+'_catalog.fits')        


        # Run SCIMES here
        res = SpectralCloudstering(d, cat, hd, criteria = ['volume','luminosity'], \
          blind = True, rms = rms, s2nlim = 3, save_all = True, user_iter=1)


        d = None

        np.savez(path+'CLUST_AFFS/'+files[f]+setup+'_params.npz', fname = files[f], \
            sigma = np.sqrt(res.escalpars[0]), sil = res.silhouette, affmat = res.affmats, clusts = res.clusters)

        os.system('rm -rf '+path+'CATALOGS/'+files[f]+setup+'_catalog.fits')
        res.catalog.write(path+'CATALOGS/'+files[f]+setup+'_catalog.fits')

        print('Save asgn cube...')
        res.clusters_asgn.writeto(path+'ASGNCUBES/'+files[f]+setup+'_asgn.fits', clobber=True)
        res.leaves_asgn.writeto(path+'LEAFASGNCUBES/'+files[f]+setup+'_leafasgn.fits', clobber=True)
        res.trunks_asgn.writeto(path+'TRUNKASGNCUBES/'+files[f]+setup+'_trunkasgn.fits', clobber=True)        

        print('Plot the tree...')
        res.showdendro(savefile=path+'TREES/'+files[f]+setup+'.png')
        res = None
        plt.close()

        print("Time:", time.time() - t)



#%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%        
#   Tag clouds on the upper and lower edges
#%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%        

if do_edgecloud:

    from skimage.morphology import square,erosion

    if os.path.isdir(mpath+'SCIMES/'+str(run)+'_RUN/EDGECUBES/') == False:
        os.system('mkdir '+mpath+'SCIMES/'+str(run)+'_RUN/EDGECUBES/') 


    fnames = np.load(mpath+'SCIMES/'+str(run)+'_RUN/SUBCUBES/subfiles.npy')

    print 'Find clouds on the edges ...'

    for f in range(len(fnames)):

        fname = fnames[f]

        cat = Table.read(mpath+'SCIMES/'+str(run)+'_RUN/CATALOGS/'+fname+'_saveall_catalog.fits')
        asgn = fits.getdata(mpath+'SCIMES/'+str(run)+'_RUN/ASGNCUBES/'+fname+'_saveall_asgn.fits')
        hd = fits.getheader(mpath+'SCIMES/'+str(run)+'_RUN/ASGNCUBES/'+fname+'_saveall_asgn.fits')
        mask = fits.getdata(mpath+'SCIMES/'+str(run)+'_RUN/SUBCUBES/'+fname+'.fits')[0,:,:]

        clusts = cat['_idx'].data

        # Make a mask with only the boarder of the data marked
        mask[np.isfinite(mask)] = 1
        mask[np.isnan(mask)] = 0
        mask2 = erosion(mask,square(3))
        mask = mask+mask2
        mask[mask == 2] = 0
        mask[0,:] = 1
        mask[:,0] = 1
        mask[-1,:] = 1
        mask[:,-1] = 1
        mcube = np.zeros(asgn.shape)
        for v in range(mcube.shape[0]):
            mcube[v,:,:] = mask

        asgn[asgn == -1] = 0
        mcube = mcube*asgn

        redge_cube = np.zeros(asgn.shape, dtype=np.int)
        redge_clouds1 = np.unique(mcube[:,:,0])
        redge_clouds2 = np.unique(mcube[:,:,mcube.shape[2]-1])    
        redge_clouds1 = redge_clouds1[redge_clouds1 != 0]
        redge_clouds2 = redge_clouds2[redge_clouds2 != 0]
        redge_clouds = redge_clouds1.tolist() + redge_clouds2.tolist()

        for i in redge_clouds:
            redge_cube[asgn == i] = i

        redge_cube = fits.PrimaryHDU(redge_cube,hd)
        redge_cube.writeto(mpath+'SCIMES/'+str(run)+'_RUN/EDGECUBES/'+fname+'_edge_asgn.fits',overwrite=True)



#%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%        
#   Find edge clouds coordinates
#%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%        

if do_edgecloudcoors:

    fnames = np.load(mpath+'SCIMES/'+str(run)+'_RUN/SUBCUBES/subfiles.npy')

    print 'Find clouds on the edges coordinates...'

    pad = 20

    r2fnames = []

    j = int(fnames[-1].split('_')[1])

    for f in range(1,len(fnames)):

        j += 1

        fname = fnames[f]

        cube1 = fits.getdata(mpath+'SCIMES/'+str(run)+'_RUN/EDGECUBES/'+fnames[f-1]+'_edge_asgn.fits')   
        cube2 = fits.getdata(mpath+'SCIMES/'+str(run)+'_RUN/EDGECUBES/'+fnames[f]+'_edge_asgn.fits')

        # Position of the sub-cubes in the full cube
        xf1 = int(fnames[f-1].split('_')[2])
        xf2 = int(fnames[f].split('_')[2])

        eclouds1 = np.unique(cube1[:,:,cube1.shape[2]-1])
        eclouds2 = np.unique(cube2[:,:,0])
        eclouds1 = eclouds1[eclouds1 != 0]
        eclouds2 = eclouds2[eclouds2 != 0]

        aeclouds1 = np.unique(cube1)
        aeclouds2 = np.unique(cube2)

        reclouds1 = list(set(aeclouds1) - set(eclouds1))
        reclouds2 = list(set(aeclouds2) - set(eclouds2))

        for i in reclouds1:
            cube1[cube1 == i] = 0

        for i in reclouds2:
            cube2[cube2 == i] = 0

        # Position of the clouds in the sub-cubes
        xc1 = np.min(np.where(cube1 != 0)[2])
        xc2 = np.max(np.where(cube2 != 0)[2])

        lcut = (xc1 - pad) + xf1  
        ucut = (xc2 + pad) + xf2 

        r2fname = 'cohrs_'+str(j)+'_'+str(lcut)+'_'+str(ucut)
        r2fnames.append(r2fname)

        jcube = np.zeros([cube1.shape[0],cube1.shape[1],cube1.shape[2]+cube2.shape[2]], dtype = np.int)
        jcube[:,:,0:cube1.shape[2]] = cube1
        jcube[:,:,cube1.shape[2]:cube1.shape[2]+cube2.shape[2]] = cube2


    np.save(mpath+'SCIMES/2_RUN/SUBCUBES/subfiles.npy', r2fnames)



# Default the recursion limit
sys.setrecursionlimit(1000)
