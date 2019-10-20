import os
import sys

import numpy as np
import time
#import aplpy

import matplotlib
import matplotlib.pyplot as plt

import cubeutils as cu

from astropy import wcs
from astropy.io import fits
from scipy.ndimage.measurements import label
from scipy.ndimage.measurements import find_objects as findobj
#from readcol import readcol

from astropy import units as u
from astropy import constants as consts
from astropy.table.table import Column
from astropy.table import Table, vstack, hstack, join
from astrodendro import Dendrogram, ppv_catalog
from astrodendro.analysis import ScalarStatistic, PPVStatistic
from skimage.morphology import convex_hull_image as hull
from skimage.measure import perimeter
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import binary_dilation

from skimage.morphology import skeletonize as sk2d

from scipy.sparse.csgraph import minimum_spanning_tree as mst
from scipy.sparse import csr_matrix
from scipy.ndimage.measurements import find_objects
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from skimage.morphology import medial_axis
from skimage.morphology import square, closing, opening

import networkx as nx
import operator

from matplotlib import cm

import matplotlib.colors as colors
import time

from pdb import set_trace as stop

from itertools import combinations




def renum_asgn(asgn, rem_clusts = None, keep_clusts = None, \
            old_bg=-1, new_bg=-1, i=0, remove = False):
  
    """
    bg: background value, default = -1
    
    i: id of the first cloud, default = 0
    """

    all_clusts = np.unique(asgn)#[1::]

    if (keep_clusts is None) & (rem_clusts is None):
        
        print("Provide cloud ids to remove or to keep")
        return asgn


    if keep_clusts is None:

        # Clusters to keep
        rem_clusts = list(rem_clusts) 
        keep_clusts = list(set(all_clusts) - set(rem_clusts))


    if rem_clusts is None:

        # Clusters to keep  
        keep_clusts = list(keep_clusts) 
        rem_clusts = list(set(all_clusts) - set(keep_clusts))


    # New idx for the cluster to keep and to remove
    if remove:
        rkeep_clusts = np.array(keep_clusts)
    else:
        rkeep_clusts = np.arange(len(keep_clusts))+i      
    
    rrem_clusts = np.zeros(len(rem_clusts), dtype=np.int)+new_bg

    # Old and new idx of all clusters
    sall_clusts = np.array([old_bg]+rem_clusts+keep_clusts)     
    rall_clusts = np.array([new_bg]+rrem_clusts.tolist()+rkeep_clusts.tolist())

    rall_clusts = rall_clusts[np.argsort(sall_clusts)]
    sall_clusts = sall_clusts[np.argsort(sall_clusts)]

    # Remake the assignment cube
    index = np.digitize(asgn.ravel(), sall_clusts, right=True)
    rasgn = rall_clusts[index].reshape(asgn.shape)

    return rasgn





def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))


def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]



def mad(x):

	y = np.median(abs(x-np.median(x)))/0.6745

	return y


def nanmad(x):

    x = x[np.isfinite(x)]
    y = np.median(abs(x-np.median(x)))/0.6745

    return y




      
mpath = './' # Main directory
pathm = mpath+'SCIMES/' # SCIMES working directory
path = pathm+'POSTPROCS/' # Postprocessing directory
path1 = pathm+'1_RUN/'
path2 = pathm+'2_RUN/'

do_edgecloud = False # Find clouds on the datacube edge and tag them
do_renasgn = False # Remove clouds on longitudinal edge of the subcube, remove them and renumber the clouds progressively
do_recmat = False # Find the overlapping clouds to keep between the two runs
do_clustrem = False # Remove the overlapping clouds and renumber everything progressively once more
do_lonlat = False # Find the Galactic coordinate positions of the clouds
do_singleasgn = False # Create a single long mask
do_fastcat = False # Generate a property catalog for the clouds, without error estimation


# Increase the recursion limit
sys.setrecursionlimit(100000)

full_cube_name = "MWISP"


#%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%        
#   Tag clouds on the upper and lower edges
#%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%        

if do_edgecloud:

  from skimage.morphology import square,erosion

  runs = [1,2]

  for run in runs:

    fnames = np.load(pathm+str(run)+'_RUN/SUBCUBES/subfiles.npy')[::-1]

    print("Find clouds on the edges ...")

    for f in range(len(fnames)):

      fname = fnames[f]

      cat = Table.read(pathm+str(run)+'_RUN/CATALOGS/'+fname+'_saveall_catalog.fits')
      asgn = fits.getdata(pathm+str(run)+'_RUN/ASGNCUBES/'+fname+'_saveall_asgn.fits')
      mask = fits.getdata(pathm+str(run)+'_RUN/SUBCUBES/'+fname+'.fits')[0,:,:]

      edges = np.zeros(len(cat))
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

      mcube = mcube*asgn

      eclusts = np.unique(mcube)
      eclusts = eclusts[eclusts != -1]
      eclusts = eclusts.astype('int')

      edges[eclusts] = 1

      cat['edge'] = edges
      os.system('rm -rf '+pathm+str(run)+'_RUN/CATALOGS/'+fname+'_edge_catalog.fits')
      cat.write(pathm+str(run)+'_RUN/CATALOGS/'+fname+'_edge_catalog.fits')	



#%&%&%&%&%&%&%&%&%&%&%&%&%&%        
#     Re-numbering asgn
#%&%&%&%&%&%&%&%&%&%&%&%&%&%        

if do_renasgn:


    runs = [1,2]
    apaths = [path1+'ASGNCUBES/', path2+'ASGNCUBES/']
    ppaths = [path1, path2]

    if os.path.isdir(path+'RCATALOGS/') == False:
        os.system('mkdir '+path+'RCATALOGS/')

    #print runfile, paths[runfile]
    #runfile = -1    
    i = 0

    for run in runs: #

        apath = apaths[run-1]

        if os.path.isdir(path+'RASGNCUBES'+str(run)+'/') == False:
            os.system('mkdir '+path+'RASGNCUBES'+str(run)+'/') 
    
        fnames = np.load(ppaths[run-1]+'SUBCUBES/subfiles.npy')#[::-1]
        

        print("Renumbering...")        

        for f in range(len(fnames)):

            fname = fnames[f]

            hdu = fits.open(apath+fname+'_saveall_asgn.fits')[0]
            data = hdu.data
            hd = hdu.header

            #rasgn = data.copy()

            aclusts = np.unique(data)[1::]

            # 1st removal: leaves on the edges
            cat = Table.read(ppaths[run-1]+'CATALOGS/'+fname+'_edge_catalog.fits')
            cat = cat[aclusts]

            # All cluster index
            all_clusts = cat['_idx'].data.tolist()

            # Leaves on the edges to remove
            leaf_on_edges = cat['_idx'].data[(cat['edge'].data == 1) & (cat['structure_type'].data == 'L')]
            #leaf_on_edges = []
	    
            if (f == 0) & (run == 1):
                clust_on_edges = np.unique(data[:,:,-1])
            elif (f == len(fnames)-1) & (run == 1):
                clust_on_edges = np.unique(data[:,:,0])
            else:
                clust_on_edges = np.unique(np.unique(data[:,:,0]).tolist() + np.unique(data[:,:,-1]).tolist())	      

            leaf_on_edges = leaf_on_edges.tolist()
            leaf_on_edges = []

            # Clusters to remove	    
            rem_clusts = np.unique(leaf_on_edges + clust_on_edges.tolist()).tolist()[1::] #remove -1 from here (background)

            # Clusters to keep	
            keep_clusts = list(set(all_clusts) - set(rem_clusts))

            # New idx for the cluster to keep and to remove
            rkeep_clusts = np.arange(len(keep_clusts))+i
            rrem_clusts = np.zeros(len(rem_clusts), dtype=np.int)-1

            # Old and new idx of all clusters
            sall_clusts = np.array([-1]+rem_clusts+keep_clusts)	    
            rall_clusts = np.array([-1]+rrem_clusts.tolist()+rkeep_clusts.tolist())

            rall_clusts = rall_clusts[np.argsort(sall_clusts)]
            sall_clusts = sall_clusts[np.argsort(sall_clusts)]

            # Remake the assignment cube
            index = np.digitize(data.ravel(), sall_clusts, right=True)
            rasgn = rall_clusts[index].reshape(data.shape)

            sall_clusts = sall_clusts[1::]
            rall_clusts = rall_clusts[1::]

            rcat = cat.copy()
            rcat['_idx'] = rall_clusts
            rcat['dendro_structure'] = sall_clusts
            rcat['orig_file'] = [fname]*len(rcat)

            rcat = rcat[rcat['_idx'].data != -1]
            rcat = rcat[np.argsort(rcat['_idx'].data)]

            _, npixs = np.unique(rasgn[rasgn != -1], return_counts=True)
            rcat['npix'] = npixs


            if f == 0:
                fcat = rcat.copy()
            else:
                fcat = join(fcat,rcat, join_type='outer')


            i = max(rkeep_clusts) + 1

            print("Total number of clusters so far:", i-1)
                          
            rasgn = fits.PrimaryHDU(rasgn,hd)
            rasgn.writeto(path+'RASGNCUBES'+str(run)+'/'+fname+'_rasgn.fits', overwrite = True)


        os.system('rm -rf '+path+'RCATALOGS/mwisp_run'+str(run)+'_rencatalog.fits')
        fcat.write(path+'RCATALOGS/mwisp_run'+str(run)+'_rencatalog.fits')

    


#%&%&%&%&%&%&%&%&%&%&%&%&%&%        
#   Recursive matching
#%&%&%&%&%&%&%&%&%&%&%&%&%&%        

if do_recmat:

    fpaths = [path1,path2]

    # Choose as the main run the one with higher average 
    # silhouette. In this case, the main run (1) becomes the
    # comparison cube. We need to understand which clouds
    # to keep in run 2.

    mrun = 1
    crun = 2

    mnames = np.load(pathm+str(mrun)+'_RUN/SUBCUBES/subfiles.npy')#[::-1]
    cnames = np.load(pathm+str(crun)+'_RUN/SUBCUBES/subfiles.npy')#[::-1]

    clcuts = []
    cucuts = []
    for cname in cnames:
        clcuts.append(np.int(cname.split('_')[2]))
        cucuts.append(np.int(cname.split('_')[3]))

    clcuts = np.asarray(clcuts)
    cucuts = np.asarray(cucuts)

    # Open the renumbered catalog
    catm = Table.read(path+'RCATALOGS/mwisp_run'+str(mrun)+'_rencatalog.fits')
    catc = Table.read(path+'RCATALOGS/mwisp_run'+str(crun)+'_rencatalog.fits')

    # Add a fake background pixels
    midxs = np.array([-1] + catm['_idx'].data.data.tolist())
    cidxs = np.array([-1] + catc['_idx'].data.data.tolist())

    mnpixs = np.array([0] + catm['npix'].data.data.tolist())
    cnpixs = np.array([0] + catc['npix'].data.data.tolist())

    # Lists contain the rows to be removed
    mremrows = []
    cremrows = []

    mpcat = catm.copy()
    cpcat = catc.copy()
        
    rmclusts = [] # mrun clusters to remove
    rcclusts = [] # mrun clusters to remove
    #smclusts = [] # mrun clusters to keep
    #scclusts = [] # crun clusters to keep

    for f in range(1,len(mnames)):
  
        smclust = []
        scclust = []

        mcube1 = fits.getdata(path+'RASGNCUBES'+str(mrun)+'/'+mnames[f-1]+'_rasgn.fits')
        mcube2 = fits.getdata(path+'RASGNCUBES'+str(mrun)+'/'+mnames[f]+'_rasgn.fits')

        mdata = np.concatenate((mcube1,mcube2), axis=2)

        mlcut = int(mnames[f-1].split('_')[2])
        mucut = int(mnames[f].split('_')[3])      

        #cname = cnames[np.where(clcuts > mlcut)][0]
        #clcut = clcuts[np.where(clcuts > mlcut)][0]
        #cucut = cucuts[np.where(cucuts < mucut)][-1]

        cname = cnames[f-1]
        clcut = int(cnames[f-1].split('_')[2])
        cucut = int(cnames[f-1].split('_')[3])

        cdata = fits.getdata(path+'RASGNCUBES'+str(crun)+'/'+cname+'_rasgn.fits')

        xsz = mcube1.shape[2]

        mcube1 = None
        mcube2 = None

        mdata = mdata[:,:,clcut-mlcut:cucut-mlcut]

        mclusts = np.unique(mdata)
        mclusts = mclusts[mclusts != -1]

        cclusts = np.unique(cdata)
        cclusts = cclusts[cclusts != -1]

        cmdata = mdata.copy()
        ccdata = cdata.copy()

        mdata = mdata.astype('int').ravel()
        cdata = cdata.astype('int').ravel()
        idx = mdata + cdata != -2
        mdata = mdata[idx]
        cdata = cdata[idx]
  
        scomp = list(set(zip(mdata.tolist(),cdata.tolist())))

        print("Find the clusters from run 1...")

        # 1st run
        for i,mclust in enumerate(mclusts):

            tups = [item for item in scomp if item[0] == mclust]
            npixs = [mnpixs[midxs == mclust][0]]	  
            clusts = [mclust]

            for t in range(len(tups)):

                tup = tups[t]

                clusts = list(clusts)
                clusts.append(tup[1])
                npixs.append(cnpixs[cidxs == tup[1]][0])	  

                clusts = np.asarray(clusts)
                npixs = np.asarray(npixs)
                npixs = npixs.tolist()
                clusts = clusts.tolist()

                clust = clusts[np.argmax(npixs)]

                if np.argmax(npixs) == 0:

                    #smclusts.append(clust)
                    smclust.append(clust)
            
                    clusts = np.asarray(clusts[1::])
                    clusts = clusts[clusts != -1]
                    #rcclusts = rcclusts + clusts.tolist()

                else:

                    #scclusts.append(clust)
                    scclust.append(clust)

        rmclusts = rmclusts + list(set(mclusts) - set(smclust))


        print("Find the clusters from run 2...")

        # 2nd run
        for i,cclust in enumerate(cclusts):

            tups = [item for item in scomp if item[1] == cclust]

            if (len(tups) == 1) & (tups[0][0] == -1):

                #scclusts.append(cclust)
                scclust.append(cclust)

        rcclusts = rcclusts + list(set(cclusts) - set(scclust))
        
	
        srmclusts = list(set(smclust) - set(np.unique(rmclusts)))
        srcclusts = list(set(scclust) - set(np.unique(rcclusts)))

        # Matching check
        ucmdata = np.unique(cmdata)[1::]
        uccdata = np.unique(ccdata)[1::]	
	
        urmdata = list(set(ucmdata)-set(srmclusts))
        urcdata = list(set(uccdata)-set(srcclusts))

        old_ucmdata = np.array([-1] + srmclusts + urmdata)
        old_uccdata = np.array([-1] + srcclusts + urcdata)

        ren_ucmdata = np.array([-1] + srmclusts + [-1]*len(urmdata))
        ren_uccdata = np.array([-1] + srcclusts + [-1]*len(urcdata))

        ren_ucmdata = ren_ucmdata[np.argsort(old_ucmdata)]
        old_ucmdata = old_ucmdata[np.argsort(old_ucmdata)]

        ren_uccdata = ren_uccdata[np.argsort(old_uccdata)]
        old_uccdata = old_uccdata[np.argsort(old_uccdata)]

        index = np.digitize(cmdata.ravel(), old_ucmdata, right=True)
        rcmdata = ren_ucmdata[index].reshape(cmdata.shape)

        index = np.digitize(ccdata.ravel(), old_uccdata, right=True)
        rccdata = ren_uccdata[index].reshape(ccdata.shape)

        urcmdata = rcmdata.copy()
        urccdata = rccdata.copy()

        urcmdata[urcmdata != -1] = 0
        urccdata[urccdata != -1] = 0

        #plt.matshow(np.nanmax(urcmdata + urccdata,axis=0),origin='lower')
        print(np.unique(urcmdata + urccdata))

        #stop()


	#stop()
    
    midxs = midxs[1::]
    cidxs = cidxs[1::]    
    
    rmclusts = np.unique(rmclusts)
    smclusts = np.array(list(set(midxs) - set(rmclusts)))
    
    rcclusts = np.unique(rcclusts)
    scclusts = np.array(list(set(cidxs) - set(rcclusts)))

    mpcat = mpcat[np.in1d(midxs,smclusts)]
    cpcat = cpcat[np.in1d(cidxs,scclusts)]
    
    os.system('rm -rf '+path+'RCATALOGS/mwisp_run'+str(mrun)+'_provcatalog.fits')
    os.system('rm -rf '+path+'RCATALOGS/mwisp_run'+str(crun)+'_provcatalog.fits')    
    
    mpcat.write(path+'RCATALOGS/mwisp_run'+str(mrun)+'_provcatalog.fits')
    cpcat.write(path+'RCATALOGS/mwisp_run'+str(crun)+'_provcatalog.fits')


#%&%&%&%&%&%&%&%&%&%&%&%&%&%        
#   Cluster removal
#%&%&%&%&%&%&%&%&%&%&%&%&%&%        

if do_clustrem:

    # First make new asgn cube with only the selected clouds
    # taken from the two new catalogs


    if os.path.isdir(path+'FINALASGNS/') == False:
    	os.system('mkdir '+path+'FINALASGNS/') 

    cat1 = Table.read(path+'RCATALOGS/mwisp_run1_provcatalog.fits')
    cat2 = Table.read(path+'RCATALOGS/mwisp_run2_provcatalog.fits')

    cat = join(cat1, cat2, join_type='outer')
    
    xin = []
    origfiles = cat['orig_file'].data.data
    for origfile in origfiles:
      xin.append(origfile.split('_')[2])
    
    xin = np.asarray(xin).astype('int')
    cat = cat[np.argsort(xin)]

    ids = cat['_idx'].data.data
    files = cat['orig_file'].data.data

    _, idx = np.unique(files, return_index=True)
    ufiles = files[np.sort(idx)]
    
    #hd = fits.getheader(pathm+'COHRS_all.fits')
    #data = np.zeros([hd['NAXIS3'],hd['NAXIS2'],hd['NAXIS1']])


    i = 0
    nids = []

    for f in range(len(ufiles)):

        print("Open file", f,"/",len(ufiles))

        name = ufiles[f]        
        asgnfile = name+'_rasgn.fits'

        x0 = int(name.split('_')[2])
        x1 = int(name.split('_')[3])

        print(name)

        if os.path.isfile(path+'RASGNCUBES1/'+asgnfile):
            hdu = fits.open(path+'RASGNCUBES1/'+asgnfile)[0]
        else:
            hdu = fits.open(path+'RASGNCUBES2/'+asgnfile)[0]
        
        asgn = hdu.data
        hd = hdu.header

        all_clusts = np.unique(asgn).tolist()
        all_clusts = all_clusts[1::]

        # Clusters to keep
        keep_clusts = ids[files == name].data.tolist()

        # Clusters to remove
        rem_clusts = list(set(all_clusts) - set(keep_clusts))

        # New idx for the cluster to keep and to remove	
        rkeep_clusts = np.arange(len(keep_clusts))+i
        rrem_clusts = np.zeros(len(rem_clusts), dtype=np.int)-1

        # Old and new idx of all clusters
        sall_clusts = np.array([-1]+rem_clusts+keep_clusts)	    
        rall_clusts = np.array([-1]+rrem_clusts.tolist()+rkeep_clusts.tolist())

        rall_clusts = rall_clusts[np.argsort(sall_clusts)]
        sall_clusts = sall_clusts[np.argsort(sall_clusts)]

        # Remake the assignment cube
        index = np.digitize(asgn.ravel(), sall_clusts, right=True)
        rasgn = rall_clusts[index].reshape(asgn.shape)

        nids = nids + rkeep_clusts.tolist()
        #nids = nids + rall_clusts.tolist()

        i = max(rkeep_clusts) + 1

        #cat['_idx'][fpos[0]:fpos[-1]] = rall_clusts

        rasgn = fits.PrimaryHDU(rasgn,hd)
        rasgn.writeto(path+'FINALASGNS/'+name+'_fasgn.fits',overwrite=True)
        
        #if f == 0:
	#  rasgn.data[rasgn.data == 0] = -2
	#rasgn.data[rasgn.data == -1] = 0
        
        #data[:,:,x0:x1] = data[:,:,x0:x1] + rasgn.data
        
        #stop()
        
    cat['_idx'] = nids    
    #cat = cat[cat['_idx'].data.data != -1]
    #cat = cat[np.argsort(cat['_idx'].data.data)]

    #cat = cat[cat['_idx'].data != -1]

    os.system('rm -rf '+path+'RCATALOGS/mwisp_semifinalcat.fits')
    cat.write(path+'RCATALOGS/mwisp_semifinalcat.fits')
    

        

#%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
# 	      Single asgn
#%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
# Create a single asgn file, by filling
# the holes of the varius asgn files

if do_singleasgn:

    ofiles = np.unique(cat['orig_file'].data.data)

    if os.path.isdir(path+'FINALSINGLEASGNS/') == False:
        os.system('mkdir '+path+'FINALSINGLEASGNS/')

    mrun = 1
    crun = 2

    mnames = np.load(pathm+str(mrun)+'_RUN/SUBCUBES/subfiles.npy')
    cnames = np.load(pathm+str(crun)+'_RUN/SUBCUBES/subfiles.npy')

    # Make full asgn cube
    
    hd = fits.getheader(mpath+full_cube_name+'.fits')
    
    peixe = fish.ProgressFish(total=len(mnames))
    print("Create full asgn cube...")

    # Connect the main cubes
    for f in range(len(mnames)):    

        peixe.animate(amount=f)

        mdata = fits.getdata(path+'FINALASGNS/'+mnames[f]+'_fasgn.fits')

        if f == 0:
	  
            data = mdata
	
        else:

	        data = np.concatenate((data,mdata),axis=2)



    data[data == 0] = -2
    data[data == -1] = 0    

    peixe = fish.ProgressFish(total=len(cnames))

    # Connect the secondary cubes
    for f in range(len(cnames)):    

        peixe.animate(amount=f)

        if cnames[f] in ofiles:

            cdata = fits.getdata(path+'FINALASGNS/'+cnames[f]+'_fasgn.fits')
            cdata[cdata == -1] = 0

        	#m0 = int(mnames[27].split('_')[2])

            c0 = int(cnames[f].split('_')[2])
            c1 = int(cnames[f].split('_')[3])	

            data[:,:,c0:c1] = data[:,:,c0:c1] + cdata
        	#data[:,:,c0-m0:c1-m0] = data[:,:,c0-m0:c1-m0] + cdata


    data[data == 0] = -1
    data[data == -2] = 0

    udata = np.unique(data)

    peixe = fish.ProgressFish(total=len(mnames))

    # Create FINALSINGLEASGNS
    for f in range(len(mnames)):    

        peixe.animate(amount=f)

        mhd = fits.getheader(path+'FINALASGNS/'+mnames[f]+'_fasgn.fits')

        m0 = int(mnames[f].split('_')[2])
        m1 = int(mnames[f].split('_')[3])	

        mdata = data[:,:,m0:m1]
        mdata = fits.PrimaryHDU(mdata,mhd)
        mdata.writeto(path+'FINALSINGLEASGNS/'+mnames[f]+'_fasgn.fits',overwrite=True)


    data = fits.PrimaryHDU(data,hd)
    data.writeto(path+'FINALSINGLEASGNS/'+full_cube_name+'_asgn.fits',overwrite=True)



#%&%&%&%&%&%&%&%&%&%&%&%&%&%        
#   Log and lat
#%&%&%&%&%&%&%&%&%&%&%&%&%&%        

if do_lonlat:

    cat = Table.read(path+'RCATALOGS/mwisp_semifinalcat.fits')
    _, idx = np.unique(cat['orig_file'].data, return_index=True)
    fnames = cat['orig_file'].data[np.sort(idx)]

    all_clusts = cat['_idx'].data
    all_fnames = cat['orig_file'].data
    all_xcoors = np.zeros(len(cat))
    all_ycoors = np.zeros(len(cat))
    all_vcoors = np.zeros(len(cat))

    print("Find leaves coordinates ...")
    peixe = fish.ProgressFish(total=len(fnames))

    for f,fname in enumerate(fnames):

        peixe.animate(amount=f)

        scat = cat[cat['orig_file'].data == fname]

        clusts = scat['_idx'].data
        xcens = np.round(scat['x_cen'].data.data).astype('int')
        ycens = np.round(scat['y_cen'].data.data).astype('int')
        vcens = np.round(scat['v_cen'].data.data).astype('int')

        hd = fits.getheader(path+'FINALASGNS/'+fname+'_fasgn.fits')
        w = wcs.WCS(hd)

        for i,clust in enumerate(clusts):

            xcoor, ycoor, vcoor = w.all_pix2world(xcens[i],ycens[i],vcens[i],0)
            vcoor = vcoor/1000.

            all_xcoors[(all_clusts == clust) & (all_fnames == fname)] = xcoor
            all_ycoors[(all_clusts == clust) & (all_fnames == fname)] = ycoor
            all_vcoors[(all_clusts == clust) & (all_fnames == fname)] = vcoor

    cat['glon_deg'] = all_xcoors*u.deg
    cat['glat_deg'] = all_ycoors*u.deg
    cat['vlsr_kms'] = all_vcoors*u.km/u.s

    os.system('rm -rf '+path+'RCATALOGS/mwisp_semifinalcat_lonlat.fits')
    cat.write(path+'RCATALOGS/mwisp_semifinalcat_lonlat.fits')




#%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%        
#  Fast physical catalog generation
#%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%        

if do_fastcat:

    fcat = Table.read(path+'RCATALOGS/mwisp_semifinalcat_lonlat.fits')
    hd = fits.getheader(path+'FINALASGNS/'+fcat['orig_file'].data[0]+'_fasgn.fits')

    cat = Table()
    for colname in fcat.colnames:
        if fcat[colname].unit:
            cat[colname] = fcat[colname].data*fcat[colname].unit
        else:
            cat[colname] = fcat[colname].data


    lab = ''

    cat['_idx'].name = 'cloud_id'

    # Find distances to outer Galaxy clouds
    glons = cat['glon_deg'].data
    vlsrs = cat['vlsr_kms'].data
    
    cat['distance_pc'] = dists*u.pc # Distances must be provided somehow

    mins = cat['minor_sigma'].data
    majs = cat['major_sigma'].data
    areas = cat['area_exact'].data
    vrmss = cat['v_rms'].data
    fluxs = cat['flux'].data
    xcoors = cat['glon_deg'].data
    ycoors = cat['glat_deg'].data

    xcoors = xcoors*np.pi/180.
    ycoors = ycoors*np.pi/180.


    deltax_pc = abs(np.pi/180.*hd['CDELT1']*dists)
    deltay_pc = abs(np.pi/180.*hd['CDELT2']*dists)
    deltav_kms = hd['CDELT3']/1000.

    # Physical constants
    mh = 1.673534*10**(-24)             # hydrogen mass CGS
    ms = 1.98900*10**33              # solar mass CGS
    pc = 3.0857*10**18                # parsec CGS   
    xco = (1/0.7)*2*10**20 # XCO(2-1) = 1.4*XCO(1-0)

    # Radius [pc]
    rads_pc = 1.91*np.sqrt((majs*deltax_pc)**2 + (mins*deltay_pc)**2)*np.sqrt(0.5)
    cat[lab+'radius_pc'] = rads_pc*u.pc

    # Velocity dispersion [km/s]
    sigs_kms = deltav_kms*vrmss
    cat[lab+'sigv_kms'] = sigs_kms*u.km/u.s

    # Scaling parameter
    scalpars_kms2_pc = sigs_kms**2/rads_pc
    cat[lab+'scalpar_kms2_pc'] = scalpars_kms2_pc*(u.km/u.s)**2/u.pc
      
    # Axis ratio
    axrats = mins/majs
    cat[lab+'pca_axis_ratio'] = axrats

    # Luminosity [K km/s pc2] and Luminosity mass [Msol]
    lcos_kkms_pc2 = fluxs*deltav_kms*deltax_pc*deltay_pc
    mlums_msun = 0.7*lcos_kkms_pc2*(xco*(2.*mh)*(1.36)*(pc*pc)/ms)
    cat[lab+'lco_kkms_pc2'] = lcos_kkms_pc2*u.K*u.km/u.s*u.pc**2
    cat[lab+'mlum_msun'] = mlums_msun*u.Msun

    # Virial mass [Msol]
    mvirs_msun = 1040*sigs_kms**2*rads_pc
    cat[lab+'mvir_msun'] = mvirs_msun*u.Msun

    # Surface brightness [K km/s]
    surbs_k_kms = lcos_kkms_pc2/(np.pi*rads_pc**2)
    cat[lab+'wco_k_kms'] = surbs_k_kms*u.K*u.km/u.s

    # Column density 
    Nh2 = surbs_k_kms*xco/areas
    cat[lab+'col_dens_cm2'] = Nh2/u.cm**2

    # Surface density [Msun/pc2]
    surds_msun_pc2 = mlums_msun/(np.pi*rads_pc**2)
    cat[lab+'surf_dens_msun_pc2'] = surds_msun_pc2*u.Msun/(u.pc**2)
          
    # Volume [pc2 km/s]
    volumes_pc2_kms = np.pi*rads_pc**2*sigs_kms
    cat[lab+'volume_pc2_kms'] = volumes_pc2_kms*u.pc**2*u.km/u.s

    # Virial parameter
    alphas = 1161*sigs_kms**2*rads_pc/mlums_msun
    cat[lab+'alpha'] = alphas

    # Volumetric density [Msun/pc3]
    dens_msun_pc3 = mlums_msun/(4/3.*np.pi*rads_pc**3)
    cat[lab+'dens_msun_pc3'] = dens_msun_pc3*u.Msun/(u.pc**3)

    # Internal pressure
    pint_kb = 1176*mlums_msun*rads_pc**(-3)*sigs_kms**2
    cat[lab+'pint_kb_k_cm3'] = pint_kb*u.K/(u.cm**3)

    # tcross [yr] (from Leroy et al. 2015)
    #rads_km = rads_pc.copy()*u.pc
    #rads_km = rads_km.to(u.km).value
    #tcross_yr = 1.23*rads_km/(np.sqrt(3)*sigs_kms)
    #tcross_yr = tcross_yr*u.s
    #tcross_yr = tcross_yr.to(u.yr).value
    #cat[lab+'tcross_yr'] = tcross_yr*u.yr

    #tdyn [yr]
    rads_km = rads_pc.copy()*u.pc
    rads_km = rads_km.to(u.km)
    tdyn_yr = rads_km/(sigs_kms*u.km/u.s)
    tdyn_yr = tdyn_yr.to(u.yr)
    cat[lab+'tdyn_yr'] = tdyn_yr

    # tff [y]
    G1 = consts.G.to(u.pc**3/(u.Msun*u.s**2))
    G1 = G1.value
    tff_yr = (3*np.pi/(32*G1*dens_msun_pc3))**0.5
    tff_yr = (tff_yr*u.s).to(u.yr).value
    cat[lab+'tff_yr'] = tff_yr*u.yr

    # Mach number (from Leroy et al. 2015)
    #mach_num = np.sqrt(3)*sigs_kms/0.45
    #cat[lab+'mach_num'] = mach_num

    # Mean volume density transfer rate
    dens_msun_pc3 = dens_msun_pc3*(u.Msun/u.pc**3)
    dens_kg_km3 = dens_msun_pc3.to(u.kg/u.km**3) 
    twoeps = dens_kg_km3*sigs_kms**3*(u.km**3/u.s**3)/rads_km
    twoeps = twoeps.to(u.erg/(u.cm**3*u.s))
    cat[lab+'twoeps_erg_cm3_s'] = twoeps

    # Energy dissipation rate
    Edis = -0.5*mlums_msun*(sigs_kms)**3/rads_pc
    Edis = Edis*u.Msun*(u.km/u.s)**3/u.pc
    Edis = Edis.to(u.Lsun)
    cat[lab+'edis_lsun'] = Edis

    # 1st Larson law inferred radius
    G2 = consts.G.to(u.km**2*u.pc/(u.Msun*u.s**2))
    G2 = G2.value
    Eh2 = 200.
    lars_rads_pc = sigs_kms**2/np.sqrt(1/3.*np.pi*G2*Eh2)
    cat[lab+'lars1_radius_pc'] = lars_rads_pc*u.pc

    # 1st Larson law inferred distance

    deltax_as = np.pi/180.*hd['CDELT1']
    deltay_as = np.pi/180.*hd['CDELT2']
    lars_dists_pc = lars_rads_pc/(1.91*np.sqrt((majs*deltax_as)**2 + (mins*deltay_as)**2))
    cat[lab+'lars1_distance_pc'] = lars_dists_pc*u.pc

    dists = dists*u.pc

    # Galactocentric and Solarcentric coordinates
    R0 = 8.1e3*u.pc
    z0 = 25.*u.pc #Reid et al. 2016
    theta = np.arcsin(z0/R0)

    xsuns = dists*np.cos(xcoors)*np.cos(ycoors)
    ysuns = dists*np.sin(xcoors)*np.cos(ycoors)
    zsuns = dists*np.sin(ycoors)

    R0cost = R0*np.cos(theta)
    R0sint = R0*np.sin(theta)

    xgals = np.full_like(xcoors,R0cost.value)*R0cost.unit - dists*(np.cos(xcoors)*np.cos(ycoors)*np.cos(theta) + np.sin(ycoors)*np.sin(theta))
    ygals = -1.*dists*np.sin(xcoors)*np.cos(ycoors)
    zgals = np.full_like(xcoors,R0sint.value)*R0sint.unit - dists*(np.cos(xcoors)*np.cos(ycoors)*np.sin(theta) - np.sin(ycoors)*np.cos(theta))

    cat[lab+'xsun_pc'] = xsuns
    cat[lab+'ysun_pc'] = ysuns
    cat[lab+'zsun_pc'] = zsuns
    cat[lab+'xgal_pc'] = xgals
    cat[lab+'ygal_pc'] = ygals
    cat[lab+'zgal_pc'] = zgals


    os.system('rm -rf '+path+'RCATALOGS/mwisp_fastphyscatalog.fits')
    cat.write(path+'RCATALOGS/mwisp_fastphyscatalog.fits')

