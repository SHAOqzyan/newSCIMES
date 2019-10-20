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
from myTree import dendroTree
import glob

from pdb import set_trace as stop

from spectral_cube import SpectralCube


from astrodendro import Dendrogram

from astropy import units as u


from astropy.wcs import WCS

from myPYTHON import *

doFITS=myFITS()

# this function is used to mask the data before using dendrogram
# to identify signal region

def maskWithFellWalker(CO12FITS,saveMaskFITS,clumpMark):
	"""

	:param CO12FITS:
	:param saveMaskFITS:
	:param clumpMark:
	:return:
	"""


	runCode="fellwalkerMWISP {} {}".format(CO12FITS,clumpMark)
	os.system(runCode)

	clumpFITS=clumpMark+"FellwalkerClump.fits"


	data,head=doFITS.readFITS(clumpFITS)

	dataCO,headCO=doFITS.readFITS(CO12FITS)

	maskData=np.zeros_like(data)

	maskData[data>0]=1

	maskedCO=maskData*dataCO

	maskedCO[maskedCO<=0]=np.nan

	#convert to 32 bit

	maskedCO=np.float32(maskedCO)
	fits.writeto(saveMaskFITS, maskedCO ,  header=head, overwrite=True)



	#saveMaskFITS, is the noise masked fits

#maskWithFellWalker("./mosaicTest2/sub11.fits", "./mosaicTest2/sub11MaskS1.fits" ,"./mosaicTest2/sub11Sigma1")


#aaaaa

def dilmasking(hdu, fchan = 8, ed = 4, th = 2,saveName=None,rms=0.5):

	data = hdu.data
	hd = hdu.header

	#free1 = data[0:fchan,:,:]
	#rms = np.nanstd(free1[np.isfinite(free1)])


	# the MWISP survey is unitform, we use a single value of RMS for 12CO data


	rmsmap = data[0]*0+ rms  #np.nanstd(free1, axis=0)



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

	#data is the file to save

	if saveName != None:
		data.writeto(saveName,overwrite=True)

	return dilmask, data, labarr

# Increase the recursion limit
sys.setrecursionlimit(100000)


# Some parameters / names to set globally

bmaj = 50./3600 # Modify with the size of the beam in deg
bmin = 50./3600 # Modify with the size of the beam in deg



class mySCIMES:
	dataPath="./data/"
	metadata = {}
	metadata['data_unit'] = u.K
	metadata['spatial_scale'] =  0.5 * u.arcmin
	metadata['beam_major'] =  50/60. * u.arcmin # FWHM
	metadata['beam_minor'] =  50/60. * u.arcmin # FWHM

	metadata['velocity_scale'] =  0.2 * u.km/u.s # FWHM

	c= 299792458.
	f=115271202000.0
	wavelength=c/f*u.meter

	metadata['wavelength'] = wavelength  # 22.9 * u.arcsec # FWHM

	sigma = 0.5 #K, noise level

	rms=sigma

	regionName=None



	subRight = "right"
	subLeft = "left"

	subUpper = "upper"

	subLower = "lower"

	#clusterSavePath = saveSCIMESPath #"./saveSCIMES/"
	fitsSuffix = "500Cluster_asgn.fits"
	#

	subRegionList=None
	subRegionDict=None


	minArea=0.02 # square degrees

	CO12FITS=None

	maskSuffix="masked.fits"

	downNoise=2.0

	def __init__(self ,regionName,CO12FITS=None):

		self.regionName=regionName

		self.subFITSPath="./{}/".format(regionName)

		if not os.path.isdir(self.subFITSPath ):
			os.system("mkdir "+self.subFITSPath  )

		self.saveSCIMESPath=  self.subFITSPath   #"./saveSCIMES/"

		self.CO12FITS=CO12FITS
		pass

	def getSubFITS(self,subRegion):
		return self.saveSCIMESPath+subRegion+".fits"

	def cropSubRegion(self,subRegion  ):
		"""
		#
		:param subRegion:
		:param subRegionD:
		:return:
		"""
		lRange,vRange= self.subregionDict[subRegion]


		print lRange,vRange

		doFITS.cropFITS( CO12FITS,Lrange=lRange ,Vrange=vRange,outFITS= self.getSubFITS(subRegion) ,overWrite=True  )





	def doSubRegion(self,subRegion,reDo=False,doMask=False,ed = 4, th = 2):
		"""
		Do dendrogram  and scimes for one sub region
		:param subRegion:
		:return:
		"""

		#first crop fits
		#self.cropSubRegion(subRegion)

		# second doDendroAnd Scimes
		subRegionFITS = self.getSubFITS(subRegion)

		processFITS=subRegionFITS

		if doMask:
			print "Masking...{} with Fellwalker".format(subRegion)

			maskedRegionFITS=self.saveSCIMESPath+subRegion+self.maskSuffix


			#hdu=fits.open(subRegionFITS)[0]

			#dilmasking(hdu,   ed = ed, th = th,saveName=maskedRegionFITS)


			maskWithFellWalker(subRegionFITS, maskedRegionFITS ,"./{}/{}".format(self.regionName,subRegion))


			processFITS=maskedRegionFITS

			self.doDendroAndScimes(  processFITS ,  calDendro=reDo, subRegion=subRegion,vScale=10 ) # no mask


		else:

			self.doDendroAndScimes(  subRegionFITS  , calDendro=reDo,subRegion=subRegion   ) # no mask







	def doDendroAndScimes(self,fitsFile,rms=0.5, minPix=500 ,reDo=False,subRegion="" ,saveAll=True,vScale=10,useVolume=True,useLuminosity=True,useVelociy=True ,calDendro=True, inputDendro=None,iinputDenroCatFile=None ):
		"""

		:param fitsFile:
		:param rms: #not important anymore
		:param minPix:
		:param reDo:
		:param subRegion: #only used to save
		:param saveAll:
		:param vScale: only useful when useVeloicty is true
		:param useVolume:
		:param useLuminosity:
		:param useVelociy:
		:return:
		"""

		if not calDendro and (inputDendro ==None or iinputDenroCatFile==None):

			print "If you do not want to calculate dendrogram, you need to provide the inputDendro and the catalog"
			return




		criteriaUsed=[]
		scales=[]
		saveMark=''

		if useLuminosity:
			criteriaUsed.append('luminosity')
			scales.append(None)
			saveMark=saveMark+'Lu'

		if useVolume:
			criteriaUsed.append('volume')

			#criteriaUsed.append('trueVolume')
			saveMark=saveMark+'Vo'

			scales.append(None)


		if useVelociy:
			#criteriaUsed.append('trueVelocity')
			criteriaUsed.append('v_rms')

			scales.append(vScale)
			saveMark=saveMark+'Ve'

		saveMark=saveMark+"{}_{}".format(minPix,vScale)

		subRegion=subRegion+saveMark

		hdu=fits.open(fitsFile)[0]
		data= hdu.data
		hd=hdu.header
		saveDendro=self.regionName+subRegion+"Dendro.fits"


		cdelt1 = abs(hd.get('CDELT1'))
		cdelt2 = abs(hd.get('CDELT2'))
		ppbeam = abs((bmaj*bmin)/(cdelt1*cdelt2)*2*np.pi/(8*np.log(2))) # Pixels per beam

		if not calDendro and os.path.isfile(  self.saveSCIMESPath+saveDendro ):
			print self.regionName," has been done, skipping..."
			return
		print("Making dendrogram...")

		#from pruning import all_true, min_vchan, min_delta, min_area
		#is_independent = all_true((min_delta(3*rms), min_area(288), min_vchan(2)))

		#is_independent = all_true( (  min_area(288) ) )

		#d = Dendrogram.compute(data, min_value=0, is_independent=is_independent, verbose=1, min_npix=10000  )
		#d = Dendrogram.compute(data, min_value=0,  verbose=1, min_npix=minPix ,min_delta=1.5 , is_independent=min_area(288) )
		catName=self.saveSCIMESPath+self.regionName+subRegion +"dendroCat.fit"


		treeFile=self.saveSCIMESPath+subRegion+"DendroTree.txt"

		if calDendro: #just for test

			d = Dendrogram.compute(data, min_value=0,  verbose=1, min_npix=minPix, min_delta=1.5 )

			d.save_to( saveDendro  )
			self.metadata["wcs"]= WCS(hd)

			cat = ppv_catalog(d, self.metadata)
			print len(cat),"<-- total number of structures?"
			#print dir(cat)

			cat.write(catName,overwrite=True)
			self.writeTreeStructure(d,treeFile)
			# add levels.
			#if saveDendro:

				#cat.write(self.saveSCIMESPath+saveDenroMark+".fit" ,overwrite=True)

				#return d,self.saveSCIMESPath+saveDenroMark+".fit"


		else: #just for test

			d =inputDendro #  do not read dendro here

			cat = Table.read(iinputDenroCatFile)
			print len(cat),"<-- total number of structures?"
			#print dir(cat)

			self.writeTreeStructure(d,treeFile)
			# add levels.



		#newVo,newVe=self.getTrueVolumeAndVrms(treeFile,iinputDenroCatFile,subRegion=subRegion)
		#cat["trueVolume"]=newVo
		#cat['trueVelocity']=newVe

		###Test, what if we multiply the volumen by the dispersion of velocity dispersion in km/s
		#cat.sort("v_rms")
		#print cat

		#print cat.colnames








		res = SpectralCloudstering(d, cat, hd, criteria = criteriaUsed  , user_scalpars=scales,  blind = True , rms = self.rms, s2nlim = 3, save_all = saveAll, user_iter=1)



		res.clusters_asgn.writeto (self.saveSCIMESPath+self.regionName+subRegion+'Cluster_asgn.fits', overwrite=True)


		clusterCat=cat[res.clusters]

		clusterCat.write( self.saveSCIMESPath  +self.regionName+subRegion +"ClusterCat.fit",overwrite=True)


	def getTrueVolumeAndVrms(self,treeFile,cat,subRegion=''):
		"""

		:param d:
		:param cat:
		:return:
		"""

		# calculate true volumes and true Vrms, by Vrms, we mean the velocity different of two leaves,which is more reasonable

		# by true volumes we mean all the sum of leaves? ,better the former


		#first get Tree

		trueVolume='trueVolume'
		trueVms='trueVms'

		vrms="v_rms"

		doTree= dendroTree(treeFile,dendroCat=cat)

		doTree.getAllLeaves(0)

		dendroTB=Table.read(cat)

		dendroTB[trueVolume]= dendroTB["flux"]
		dendroTB[trueVms]= dendroTB["v_rms"]

		for eachRow in  dendroTB:
			allL=doTree.getAllLeaves(eachRow["_idx"] )

			twoChildren=doTree.getChildren(  eachRow["_idx"] )



			tbSubLeaves=dendroTB[allL]


			if len( tbSubLeaves)>1:

				trueDV=  max(tbSubLeaves['v_cen'])- min(tbSubLeaves['v_cen'])
				trueDV=trueDV/1000

			else:
				trueDV=  tbSubLeaves[0][vrms]
				trueDV=trueDV

			#print trueDV,len(tbSubLeaves )
			#tbSubChildren=dendroTB[twoChildren]

			eachRow[trueVolume] =  eachRow["area_exact"]* trueDV

			eachRow[trueVolume] =  eachRow["area_exact"]* trueDV
			eachRow[trueVms] = trueDV




			#allC= tbSubChildren["area_exact"]*tbSubChildren["v_rms"]

			#print np.sum(allC)



		return dendroTB[trueVolume], dendroTB[trueVms]




	def doScimes(self,dendroFITS,hd ):


		self.metadata["wcs"]= WCS(hd)
		d = Dendrogram.load_from( dendroFITS   )
		cat = ppv_catalog(d, self.metadata)
		res = SpectralCloudstering(d, cat, hd, criteria = ['volume' ],   blind = True, rms = self.rms, s2nlim = 3, save_all = True, user_iter=1)

		res.clusters_asgn.writeto (self.regionName+'Cluster_asgn.fits', overwrite=True)
		#save the catalog


	def getLVrange(self, CO12FITS ):
		data,head=myFITS.readFITS(CO12FITS)
		wcsCO=WCS(head )

		Nz,Ny,Nx=data.shape

		l0,_,v0= wcsCO.wcs_pix2world(0,0,0,0)

		l1,_,v1= wcsCO.wcs_pix2world(Nx-1,Ny-1,Nz-1,0)

		v0,v1=v0/1000.,v1/1000

		return [l0,l1],[v0,v1]



	def getDivideCubesName(self,CO12FITS, divideConfig ):
		"""

		velocity is in km/s

		longitude is in degree

		Perticularly used for WMSIPS
		:param CO12FITS:
		:return:
		"""

		# produces subscube names that their corresponding vRange, and Lrange, which would be used to crop the cube

		#

		startL = divideConfig["startL"]
		lStep= divideConfig["lStep"]
		lSize = divideConfig["lSize"]
		startV = divideConfig["startV"]
		vStep =  divideConfig["vStep"]
		vSize = divideConfig["vSize"]


		data,head=myFITS.readFITS(CO12FITS)

		#find the, largest l and minimum v,

		#find the minimum l and maximum v,
		wcsCO=WCS(head )

		Nz,Ny,Nx=data.shape

		l0,_,v0= wcsCO.wcs_pix2world(0,0,0,0)

		l1,_,v1= wcsCO.wcs_pix2world(Nx-1,Ny-1,Nz-1,0)

		v0,v1=v0/1000.,v1/1000



		NL= int(  abs(l1-l0)/lStep ) +1
		NV= int( abs(v1-v0)/vStep ) +1

		regionList=[]

		regionDict={}



		for i in range(NL):
			for j in range(NV):

				countL=i+1
				countV=j+1

				regionName="sub{}{}".format(countL,countV)
				if i==0:
					lRange=[l0,startL-lSize]
				else:
					lRange=[startL-lStep*i,startL-lStep*i-lSize]

				if j==0:
					vRange=[v0,startV+vSize ]

				else:
					vRange=[startV+vStep*j,startV+vStep*j+vSize ]

				regionDict[ regionName ] = [ lRange,vRange ]
				regionList.append(regionName)


				if  startV+vStep*j+vSize>v1:
					break

			if  startL-lStep*i-lSize<l1:
				break

		self.subRegionList=regionList
		self.subregionDict= regionDict

		return regionList, regionDict

	def divdeCubes(self,CO12FITS,regionList,regionDict ):
		"""
		### # # # # # #
		:param regionList:
		:param regionDict:
		:return:
		"""
		savePath=self.saveSCIMESPath

		#os.system("mkdir "+savePath )

		for eachRegion in regionList:
			lRange,vRange=regionDict[eachRegion]
			print "producing  ", eachRegion+".fits"
			doFITS.cropFITS( CO12FITS,Lrange=lRange ,Vrange=vRange,outFITS=savePath+eachRegion+".fits"  )



	def produceRunSH(self, CO12FITS, regionList, regionDict  ): #
		"""
		The reason to use sh file is to guareette the memeory would be released when one cube is done

		:param CO12FITS:
		:param regionList:
		:param regionDict:
		:return:
		"""

		print "Producing sh script file..."
		f = open("runSubcube.sh", "w")


		for eachRegion in regionList:

			runCommand= "python myScimes.py "  +  eachRegion
			f.write(  runCommand +"\n")
		f.close()

	def getUpperCube(self,cubeName,cubeList):


		lIndex,vIndex = cubeName[3:5]
		restNameStr= cubeName[0:3]
		upperCube=restNameStr+"{}{}".format(int(lIndex), int(vIndex)+1)

		if upperCube in cubeName:
			return upperCube

		return None


	def getNearCube(self,cubeName,cubeList, position="right"):

		lIndex,vIndex = cubeName[3:5]
		restNameStr= cubeName[0:3]



		if  position== self.subRight:
			nextCube=restNameStr + "{}{}".format(int(lIndex)+1, int(vIndex) )

		if  position== self.subLeft:
			nextCube=restNameStr + "{}{}".format(int(lIndex)-1, int(vIndex) )

		if  position== self.subUpper:
			nextCube=restNameStr + "{}{}".format(int(lIndex) , int(vIndex)+1 )

		if  position== self.subLower:
			nextCube=restNameStr + "{}{}".format(int(lIndex) , int(vIndex)-1 )


		#print nextCube
		if nextCube in cubeList:
			return nextCube

		return None





	def combinSubCubes(self,CO12FITSAll,cubeList):
		"""
		This function is used to combine the subcubes into a common one
		:param CO12FITS:
		:param cubeList:
		:return:
		"""

		# this function is essentional

		#Presumably, the cubes are connected

		#first, find out the lrange and vrange of those cubes, cut out the

		# conbine the cluster assign, and cluster catalog
		print "Combing all sub cubes?"
		clusterCubePath=self.saveSCIMESPath #"/home/qzyan/WORK/myDownloads/testScimes/saveSCIMES/"
		fitsSuffix = "500Cluster_asgn.fits"


		# find the Lrange, and vRange, for the cubes

		leftLs=[]
		rightLs= []

		upperV=[]
		lowerV=[]


		for eachSub in cubeList:
			clusterCube = eachSub+fitsSuffix

			lRange,vRange=self.getLVrange( clusterCubePath+clusterCube )


			leftLs.append(  max(lRange) )
			rightLs.append( min(lRange) )

			upperV.append(  max(vRange) )
			lowerV.append( min(vRange) )



		lRange,vRange= [max( leftLs), min(rightLs )], [min(lowerV ), max( upperV)  ]



		#crop the CO12FITS
		mosaicFITScrop= "mosaicFITScrop.fits"

		if 1:
			doFITS.cropFITS(CO12FITSAll,Lrange=lRange,Vrange=vRange,outFITS= mosaicFITScrop,overWrite=True)

		mosaicData,mosaicHead=myFITS.readFITS(mosaicFITScrop)

		mosaicCluster=np.zeros_like( mosaicData )
		wcsMosaic=WCS(mosaicHead)

		idCol= "_idx"

		modelTB=Table.read( clusterCubePath+cubeList[0]+"500ClusterCat.fit" )

		acceptCloudTB= Table(  modelTB[0]  )

		acceptCloudTB.remove_row(0)



		for eachSub in cubeList:
			clusterCube = clusterCubePath+eachSub+fitsSuffix

			subData,subHead= myFITS.readFITS( clusterCube  )

			clusterCat= clusterCubePath+eachSub+"500ClusterCat.fit"

			clusterCatTB=Table.read( clusterCat )

			leftRegion = self.getNearCube(eachSub, cubeList,position=  self.subLeft )
			rightRegion =self.getNearCube(eachSub, cubeList,position=  self.subRight )
			lowerRegion = self.getNearCube(eachSub, cubeList,position= self.subLower )
			upperRegion  = self.getNearCube(eachSub, cubeList,position=self.subUpper )


			totalCloudReject= []

			if leftRegion !=None:

				# find those touches left

				cutleft=subData[ :, :, 0  ]

				cutleft=cutleft.reshape(-1)

				totalCloudReject=np.concatenate(  [ totalCloudReject, cutleft  ] )
				#cloudsTouchLeft=set(cutleft )

				#print cloudsTouchLeft

			if rightRegion !=None:
				cutRight=subData[ :, :, -1 ]

				cutRight=cutRight.reshape(-1)

				totalCloudReject=np.concatenate(  [ totalCloudReject, cutRight  ] )



			if lowerRegion !=None:
				cutLower=subData[ 0, :, :  ]

				cutLower=cutLower.reshape(-1)
				totalCloudReject=np.concatenate(  [ totalCloudReject, cutLower  ] )

			if upperRegion !=None:
				cutUpper=subData[ -1, :,: ]

				cutUpper=cutUpper.reshape(-1)

				totalCloudReject=np.concatenate(  [ totalCloudReject, cutUpper  ] )

			cloudsTobeRemove= map(int, set( totalCloudReject) )
			#



			##
			lastTwoDigital=int( eachSub[-2:] )

			#get lbv0Index
			wcsSub=WCS(subHead) # wcsMosaic
			l0,b0,v0 =   wcsSub.wcs_pix2world(0,0,0,0)
			l0Index,b0Index,v0Index= wcsMosaic.wcs_world2pix( l0,b0,v0,0 )
			l0Index,b0Index,v0Index= map(  int, 		[l0Index,b0Index,v0Index ] )
			lbv0MergeIndex=[ l0Index,b0Index,v0Index ]

			for eachCluster in clusterCatTB:
				clusterID= eachCluster[idCol ]

				if  clusterID in cloudsTobeRemove:
					continue

				#Examine area

				if eachCluster["area_exact"]/3600./3600. < self.minArea: # in square degress
					print "{} of {} is too small, rejected.".format(clusterID ,eachSub )
					continue


				if self.isDubplicated(eachCluster,acceptCloudTB):
					print " {} of {} is duplicated! ignore...".format(clusterID ,eachSub )

					continue

				else:
					acceptCloudTB.add_row(eachCluster)



				newIndex=clusterID*100+lastTwoDigital
				eachCluster[idCol] = newIndex
				acceptCloudTB.add_row( eachCluster  )
				# merge this cloud into the old one

				#

				self.mergeSub(clusterID, newIndex,subData,  mosaicCluster, lbv0MergeIndex,acceptCloudTB )


			#save for test

		fits.writeto("mergedCluster.fits",mosaicCluster,mosaicHead,overwrite=True )
		acceptCloudTB.write("mergeCat.fit" , overwrite=True )

	def touchEdge(self,subID,subFITS):
		"""
		check if a cloud touches the edge of cube, any edge
		:param subID:
		:param subFITS:
		:return:
		"""

		data,head=myFITS.readFITS(subFITS)

		Nz,Ny,Nx= data.shape
		Zs,Ys,Xs=np.where(data==subID)

		if max(Zs) == Nz-1 or min(Zs)==0:
			return True

		if max(Ys) == Ny-1 or min(Ys)==0:
			return True

		if max(Xs) == Nx-1 or min(Xs)==0:
			return True



		return False







	def mergeSub(self,subID, newID, subData, mergeData,lvb0MergeIndex, acceptCloudTB, force = True ):

		"""
		Merge sub cluster into mergedData with newID
		:param subID:
		:param subData:
		:param subHead:
		:param mergeData:
		:param mergeHead:
		:return:
		"""
		# important function
		#
		#find the index of subID in subData, convertID

		print "merging...{} of sub{}".format( subID, str(newID)[-2:]  )


		#l0,b0,v0 = lbv0  # wcsSub.wcs_pix2world(0,0,0,0)

		#################

		#l0Index,b0Index,v0Index= wcsMerge.wcs_world2pix( l0,b0,v0,0 )
		l0Index,b0Index,v0Index= lvb0MergeIndex #map(  int, 		[l0Index,b0Index,v0Index ] )

		#### ####

		subIndex=np.where( subData==subID )

		vIndex=subIndex[0]+v0Index

		bIndex=subIndex[1]+b0Index
		lIndex=subIndex[2]+l0Index

		combIndex=tuple([vIndex,bIndex,lIndex])



		if np.sum( mergeData[ combIndex] )== 0:

			mergeData[ combIndex] =newID

			return
		else:

			#save to clouds
			existClouds= mergeData[ combIndex ]
			existClouds=existClouds.reshape(-1)

			existClouds= list( set( existClouds) )


			existClouds= np.array(  map(int, existClouds )  )

			existClouds=existClouds[existClouds>0]

			#print existClouds, "??????????????????????????????"

			#clashC=len(   set( existClouds)  )


			clashC=len(   existClouds   )

			# If there is only one cloud in clash, and one of them touches the edge, keep the one that do not touch edge

			# if

			print existClouds,"?????????????????"
			print clashC,"??????????????????????"

			if  clashC ==1: #

				firstID= int( existClouds[0] )


				preRegionFITS= self.saveSCIMESPath+"sub"+str(firstID)[-2:]+self.fitsSuffix

				preID= int(  str(firstID)[0:-2 ] )

				preTouchEdge= self.touchEdge(preID, preRegionFITS)
				# check if the latter one touches edges

				secondID = int( newID )
				latterRegionFITS = self.saveSCIMESPath+"sub"+str(secondID)[-2:]+self.fitsSuffix
				latterTouchEdge = self.touchEdge(subID, latterRegionFITS)

				#
				if  preTouchEdge and not latterTouchEdge:

					print "Keeping the new one: ", newID

					#remove the old one from acceptCloudTB
					self.removeByID(firstID, acceptCloudTB)
					mergeData[ np.where(mergeData==firstID) ] = 0

					# keep the latter one
					mergeData[ combIndex] =newID


					return


				
				if force:
					
					print "Forcing .... Keeping the new one: ", newID

					#remove the old one from acceptCloudTB
					self.removeByID(firstID, acceptCloudTB)
					mergeData[ np.where(mergeData==firstID) ] = 0

					# keep the latter one
					mergeData[ combIndex] =newID


					return



				if  not preTouchEdge and  latterTouchEdge:
					print "Keeping the Old one", firstID
					#remove the new one from acceptCloudTB
					self.removeByID(newID, acceptCloudTB)

					return
				if  preTouchEdge and latterTouchEdge:

					#both would be replaed, keed the old one temperately
					print "Both touches edges... Keeping the Old one", firstID
					#remove the new one from acceptCloudTB
					self.removeByID(newID, acceptCloudTB)

					return

					#return preTouchEdge, latterTouchEdge,"Former and later touch? "

				#print "both "

				assert 1 == 0, "Two independed clouds ({}, {}) clash...  This is a severe problem!!!".format( firstID, newID )


			if clashC>1:# more than one cloud found #But, what if all of then touches edges?
				allEdge=True
				for eachClashC in existClouds:
					preRegionFITS= self.saveSCIMESPath+"sub"+str(eachClashC)[-2:]+self.fitsSuffix

					preID= int(  str(eachClashC)[0:-2 ] )

					if not self.touchEdge(preID, preRegionFITS):
						allEdge=False
				if allEdge or force:# if all the old cluster touches edges, replaces them with the new one
					if not allEdge and force: # if force merge
						print "Force merging ... ",newID
					# #
					#remove old clouds
					for eachClashC in existClouds:

						mergeData[ np.where(mergeData==eachClashC) ] = 0


						self.removeByID(eachClashC, acceptCloudTB)

					mergeData[ combIndex ] = newID

					return




			else:



				print set( existClouds),"???????????????Existing?????????????"
				print subID, "to merge"

				assert np.sum( mergeData[ combIndex] ) == 0, "More than one Clouds overlapping...  This is a severe problem!!!"



	# #


	def removeByID(self,removeID,acceptCloudTB):
		"""

		:param removeID:
		:param acceptCloudTB:
		:return:
		"""
		rmIndex= np.where( acceptCloudTB["_idx"]==removeID)[0][0]

		acceptCloudTB.remove_row( rmIndex )



	def isDubplicated(self,testRow,acceptCloudTB):
		"""
		###########
		:param testRow:
		:param acceptCloudTB:
		:return:
		"""

		############
		if len(acceptCloudTB)==0:
			return False
		l1,b1,v1 = self.getLBVFromRow( testRow )
		radius1=testRow["radius"]

		allL=acceptCloudTB["x_cen"]
		allb=acceptCloudTB["y_cen"]
		allv=acceptCloudTB["v_cen"]/1000.

		allRadius=acceptCloudTB["radius"]



		distances=  (allL-l1)**2+  (allb-b1)**2 + ( allv-v1 )**2+ ( allRadius-radius1 )**2
		distances=np.sqrt( distances )


		if  min(distances)<0.01:
				return True

		return False



	def getLBVFromRow(self, row):
		"""

		:param row:
		:return:
		"""
		l=row["x_cen"]
		b=row["y_cen"]
		v=row["v_cen"]/1000.

		return l,b,v



	def getLEdgeType(self,intData  ):
		"""
		#
		:param Data:
		:param subHead:
		:return:
		"""
		touchLeft=False
		touchRight= False

		###
		leftSum = np.sum( intData[:, 0] )


		rightSum = np.sum( intData[:, -1 ] )



		if leftSum  >  0:
			touchLeft=True

		if rightSum  >  0:
			touchRight=True


		return [ touchLeft ,  touchRight ]

	def getVEdgeType(self,lvData  ):
		"""

		:param Data:
		:param subHead:
		:return:
		"""

		touchUpper=False
		touchLower= False


		upperSum = np.sum( lvData[-1,: ] )

		lowerSum = np.sum( lvData[0, : ] )


		if upperSum > 0:
			touchUpper=True

		if lowerSum > 0:
			touchLower=True

		return [ touchLower , touchUpper  ]


	def getLostClouds(self,cloudMaskFITS,CO12FITS,saveName="leftOutCO.fits"):
		"""
		Produce a fits of subtraction of cloudMaskFITS  and CO12FITS to examine the CO emission left out
		# this happens when there is a outflow, and somehow, cloud touches edges in all subcubes
		:param cloudMaskFits:
		:param CO12FITS:
		:return:
		"""
		#assuming they are equal
		##
		dataCloud,headCloud= myFITS.readFITS(cloudMaskFITS)

		#hduCO12=fits.open(CO12FITS)[0]
		#createCO mask
		#maskedFITS="maskedCO12.fits"
		#dilmasking(hduCO12,saveName=maskedFITS,th=3)


		dataCO,headCO=  myFITS.readFITS(CO12FITS )

		dataCO[  dataCO< 0.15   ]=0

		dataCO[  np.where( dataCloud>0) ]=0

		fits.writeto(saveName,  dataCO  , header=headCO, overwrite=True)


	def addCloud(self,subID,subRegion, mosaicData, mosaicHead,acceptCloudTB ,force=False):
		"""
		add lefted large molecular clouds

		#

		:param subID:
		:param subClusterFITS:
		:param conData:
		:param conHead:
		:param conCatFile:
		:return:
		"""
		subClusterFITS=self.saveSCIMESPath+subRegion+self.fitsSuffix
		# step1,
		subData,subHead=myFITS.readFITS( subClusterFITS )
		#merge..
		#
		wcsMosaic=WCS(mosaicHead)
		newID=subID*100+int(subRegion[-2:])

		wcsSub=WCS(subHead) # wcsMosaic
		l0,b0,v0 =   wcsSub.wcs_pix2world(0,0,0,0)
		l0Index,b0Index,v0Index= wcsMosaic.wcs_world2pix( l0,b0,v0,0 )
		l0Index,b0Index,v0Index= map(  int, 		[l0Index,b0Index,v0Index ] )
		lbv0MergeIndex=[ l0Index,b0Index,v0Index ]





		self.mergeSub( subID, newID, subData, mosaicData,lbv0MergeIndex, acceptCloudTB, force=force)



	def writeTreeStructure(self,dendro,saveName):

		f=open( saveName,'w')

		#for eachC in self.dendroData:
		for eachC in dendro:

			parentID=-1

			p=eachC.parent

			if p!=None:

				parentID=p.idx

			fileRow="{} {}".format(eachC.idx,parentID)
			f.write(fileRow+" \n")

		f.close()

	def ZZZZ(self):
		pass










testConfigs={}
mT1={}


mT1["startL"] = 50
mT1["lStep"] = 10
mT1["lSize"] = 10
mT1["startV"] = 0
mT1["vStep"] = 40
mT1["vSize"] = 40

testConfigs["mosaicTest1"]= mT1

mT2={}


mT2["startL"] = 50
mT2["lStep"] = 5
mT2["lSize"] = 10
mT2["startV"] = 0
mT2["vStep"] = 10
mT2["vSize"] = 40

testConfigs["mosaicTest2"]= mT2


import argparse

parser = argparse.ArgumentParser(description='This is a demo script for test cloud identification.')
parser.add_argument('-ra','--runAll', help='run all',required=False,action='store_true')
parser.add_argument('-r1','--runOne', help='run one sub region',required=False,action='store_true' )

parser.add_argument('-n','--normal', help='run code normally',required=False,action='store_true' )



parser.add_argument('-t','--testCode', help='test code.',required=False)
parser.add_argument('-s','--subRegion', help='Subregion of test code.',required=False)
parser.add_argument('-r','--redo', help='redo the region.',required=False, action='store_true' )
parser.add_argument('-m','--mask', help='mask the region.',required=False, action='store_true' )

parser.add_argument('-v','--volume', help='mask the region.',required=False, action='store_true' )
parser.add_argument('-d','--divide', help='divide the cube.',required=False, action='store_true' )



args = parser.parse_args()
print args

if args.runOne : #run dendrogram and SCIMES on a single file
	##
	CO12FITS="G2650CO12.fits" # test regions
	#expecting  test code, and test subregion
	if args.testCode==None or args.subRegion==None:
		print "Test region and subRegion are needed...quiting..."
		sys.exit()

	else:
		print "Doing {} of {} ".format(args.subRegion,args.testCode)
		if args.redo:
			print "Redoing...{}".format( args.subRegion )

		#
		doScimes=mySCIMES( args.testCode )
		doScimes.CO12FITS=CO12FITS
		regionList,regionDict= doScimes.getDivideCubesName( CO12FITS, testConfigs[ args.testCode] )

		doScimes.doSubRegion("sub11",ed=4,th=2,doMask=args.mask,reDo=args.redo)

if args.runAll: # Pipline for run all
	doScimes=mySCIMES( args.testCode )
	doScimes.CO12FITS="G2650CO12.fits"



	#first re dividing cubes
	print "Divding the whole data cube according to the configuration"

	configTest=testConfigs[doScimes.regionName]

	#======================================================


	regionList,regionDict= doScimes.getDivideCubesName( doScimes.CO12FITS,configTest )

	if args.divide:
		doScimes.divdeCubes(doScimes.CO12FITS,regionList,regionDict)

	#done

	#=====================================================


	for eachSub in regionList:
		if eachSub in ['sub11','sub12','sub13']:
			continue
		doScimes.doSubRegion(eachSub,ed=4,th=2,doMask=args.mask,reDo=args.redo)




#python myScimes.py -r1 -t mosaicTest1 -s sub11 -r -m
#python myScimes.py -ra -t mosaicTest1   -r -m #run al for mosaicTest1 -s, and redo the procedure with mask


#######################################
else:
###
	print "run code without any prameters..."

	# test G214. to see the principle of SCIMES
	#
	#regionName="G214CO12"
	#regionName="test2Small"


	#regionName="complicatedTest"
	#regionName="boundaryTest"


	CO12FITS=regionName+".fits"
	doScimes=mySCIMES( regionName )
	doScimes.CO12FITS=CO12FITS

	hdu=fits.open(CO12FITS)[0]




	maskFITS="./{}/{}masked.fits".format(regionName,regionName)
	dilmasking(hdu, ed = 4, th = 2,saveName=maskFITS,rms=0.5)

	#parameter contral

	#test the best  parameters
	#volume lost weak emissions,

	#for minPixels in [100,200,500,1000 ]:
	for minPixels in [ 500 ]:

		#minPixels=500

		#doScimes.doDendroAndScimes( maskFITS,reDo=True, saveAll=False , minPix=minPixels,vScale=vscale, useLuminosity=True, useVolume=False,useVelociy=True, subRegion="LuVeSingleTestPureCluster" )

		saveAll=True

		dendroMark="dendroSave{}".format(minPixels)


		#dendroINPUT,catName=doScimes.doDendroAndScimes( maskFITS,reDo=True, saveAll=saveAll , minPix=minPixels, saveDendro=True,saveDenroMark=dendroMark, useLuminosity=False, useVolume=True,useVelociy=False, calDendro=True  )

		catName="./{}/dendroSave{}.fit".format(regionName,minPixels)
		dendroName="./{}/dendroSave{}.fits".format(regionName,minPixels)

		print dendroName
		dendroINPUT=  Dendrogram.load_from(dendroName)



		#for vscale in [16,18,20,22,24,26,28,30]:
			#single criteria does not work well
			#doScimes.doDendroAndScimes( maskFITS,reDo=True, saveAll=saveAll ,vScale=vscale, minPix=minPixels, useLuminosity=True, useVolume=True,useVelociy=True,  calDendro=False, inputDendro=dendroINPUT,iinputDenroCatFile=catName)


			#doScimes.doDendroAndScimes( maskFITS,reDo=True, saveAll=saveAll ,vScale=vscale, minPix=minPixels, useLuminosity=False, useVolume=False,useVelociy=True,  calDendro=False,inputDendro=dendroINPUT,iinputDenroCatFile=catName)

			#doScimes.doDendroAndScimes( maskFITS,reDo=True, saveAll=saveAll ,vScale=vscale, minPix=minPixels, useLuminosity=True, useVolume=False,useVelociy=False,  calDendro=False,inputDendro=dendroINPUT,iinputDenroCatFile=catName)

			#doScimes.doDendroAndScimes( maskFITS,reDo=True, saveAll=saveAll ,vScale=vscale, minPix=minPixels, useLuminosity=False, useVolume=True,useVelociy=False,   calDendro=False,inputDendro=dendroINPUT,iinputDenroCatFile=catName)


			# we have to add velocity

 			#doScimes.doDendroAndScimes( maskFITS,reDo=True, saveAll=saveAll ,vScale=vscale, minPix=minPixels, useLuminosity=True, useVolume=False,useVelociy=True, calDendro=False, inputDendro=dendroINPUT,iinputDenroCatFile=catName)


			#doScimes.doDendroAndScimes( maskFITS,reDo=True, saveAll=saveAll ,vScale=vscale, minPix=minPixels, useLuminosity=True, useVolume=True,useVelociy=False, calDendro=False, inputDendro=dendroINPUT,iinputDenroCatFile=catName)
			#doScimes.doDendroAndScimes( maskFITS,reDo=True, saveAll=saveAll ,vScale=vscale, minPix=minPixels, useLuminosity=False, useVolume=True,useVelociy=True,calDendro=False, inputDendro=dendroINPUT,iinputDenroCatFile=catName)




	if 0:

		doScimes.doDendroAndScimes( maskFITS,reDo=True, saveAll=saveAll , minPix=minPixels, useLuminosity=True, useVolume=False,useVelociy=True,  )
		doScimes.doDendroAndScimes( maskFITS,reDo=True, saveAll=saveAll , minPix=minPixels, useLuminosity=True, useVolume=False,useVelociy=False,  )


		doScimes.doDendroAndScimes( maskFITS,reDo=True, saveAll=saveAll , minPix=minPixels, useLuminosity=False, useVolume=True,useVelociy=False,   )
		doScimes.doDendroAndScimes( maskFITS,reDo=True, saveAll=saveAll , minPix=minPixels, useLuminosity=False, useVolume=False,useVelociy=True,   )


		doScimes.doDendroAndScimes( maskFITS,reDo=True, saveAll=saveAll , minPix=minPixels, useLuminosity=True, useVolume=True,useVelociy=False,  )
		doScimes.doDendroAndScimes( maskFITS,reDo=True, saveAll=saveAll , minPix=minPixels, useLuminosity=False, useVolume=True,useVelociy=True,  )

		doScimes.doDendroAndScimes( maskFITS,reDo=True, saveAll=saveAll , minPix=minPixels, useLuminosity=True, useVolume=True,useVelociy=True,   )





	#mask





sys.exit()






#manipulate the prameters


if 0: # test

	#
	#
	region="G2650CO12"
	CO12FITS=region+".fits"
	doScimes=mySCIMES( region )

	subFITSPath="./mosaicTest1/"

	saveSCIMESPath="./mosaicTest1Save/"
	regionList,regionDict=doScimes.getDivideCubesName( CO12FITS, startL=50,lStep=10,lSize=10,startV=0,vStep=40,vSize=40 )

	#doScimes.divdeCubes( CO12FITS,regionList,regionDict,savePath="./mosaicTest1/")
	doScimes.produceRunSH(CO12FITS, regionList, regionDict )


if 0: # test divide sub cubes

	#first test on overlapping, as do the cube the second time, like colobo
	#
	#
	#
	# mosaicTest1, no overlap test
	region="G2650CO12"
	CO12FITS=region+".fits"
	doScimes=mySCIMES( region )

	regionList,regionDict=doScimes.getDivideCubesName( CO12FITS, startL=50,lStep=10,lSize=10,startV=0,vStep=40,vSize=40 )

	#doScimes.divdeCubes( CO12FITS,regionList,regionDict,savePath="./mosaicTest1/")



if 0:# get back left out large size molecular cloud
	# because there is no decent way to further decompose those clouds, we simply add the clouds that is mostly covered clouds
	# pass
	region="left1Cut"
	doScimes=mySCIMES( region )
	#add first

	acceptCloudTB=Table.read("mergeCat.fit")
	mosaicData,mosaicHead =myFITS.readFITS("mergedCluster.fits")


	doScimes.addCloud(  1488,"sub22",mosaicData,    mosaicHead,acceptCloudTB,force=True)

	#doScimes.addCloud(  899,"sub23",mosaicData,    mosaicHead,acceptCloudTB)


	#save the fits
	fits.writeto("finalCube.fits",mosaicData, header=mosaicHead,overwrite=True)
	acceptCloudTB.write("finalCat.fits" ,overwrite=True)





if 0:#  to see how many cloud are left

	region="left1Cut"
	doScimes=mySCIMES( region )

	doScimes.getLostClouds("finalCube.fits","/home/qzyan/WORK/dataDisk/MWISP/G2650/merge/G2650CO12.fits")



if 0:#  to see how many cloud are left

	region="left1Cut"
	doScimes=mySCIMES( region )

	doScimes.getLostClouds("mergedCluster.fits","/home/qzyan/WORK/dataDisk/MWISP/G2650/merge/G2650CO12.fits",saveName="leftOutTest1")



if 0: # for
	region="left1Cut"

	CO12FITSAll= "/home/qzyan/WORK/dataDisk/MWISP/G2650/merge/G2650CO12.fits"

	doScimes=mySCIMES( region )

	cubeList=["sub11","sub12"   ]

	doScimes.combinSubCubes(CO12FITSAll,cubeList )



# for server
if 0:

	region="left1Cut"
	doScimes=mySCIMES( region )
	CO12FITS=region+".fits"
	maskedFITS=region+"MaskedData.fits"
	dendroFITS=region+"Dendro.fits"
	#do mask
	#
	hdu=fits.open(CO12FITS)[0]
	dilmasking(hdu,saveName=maskedFITS,th=3)

	doScimes.doDendroAndScimes(maskedFITS)


	#regionList, regionDict=doScimes.getDivideCubesName(CO12FITS)
	#doScimes.produceRunSH(CO12FITS, regionList, regionDict )



#region="v1Cut"
#region='sub1'
#region='sub21'
#region="G214CO12"

# get parameters
if 0:
	region="G2650CO12"
	doScimes=mySCIMES( region )
	CO12FITS=region+".fits"

	regionList, regionDict=doScimes.getDivideCubesName(CO12FITS)
	doScimes.produceRunSH(CO12FITS, regionList, regionDict )



	#doScimes.divdeCubes(CO12FITS, regionList, regionDict )

if 0: # run cube

	#pass
	if  len(sys.argv) < 2:
		print "Not enouth parameters, doing others..."

		########################

		region="left1Cut"

		CO12FITSAll= "/home/qzyan/WORK/dataDisk/MWISP/G2650/merge/G2650CO12.fits"

		doScimes=mySCIMES( region )

		doScimes.subFITSPath="./mosaicTest1/"

		doScimes.saveSCIMESPath="./mosaicTest1Save/"


		#producd list
		imax=3
		jmax=3

		cubeList=[]
		for i in range(4):
			if i+1>imax:
				break
			for j in range(9):
				if j+1>jmax:
					break
				regionI=i+1
				regionJ=j+1
				tiles="sub{}{}".format(regionI, regionJ)
				cubeList.append(tiles)

		print  "Doing list: " ,cubeList
		#aaaa


		#cubeList=["sub11", "sub12" , "sub13", "sub14", "sub15", "sub16" , "sub17", "sub18",  "sub19",   "sub21" , "sub22", "sub23", "sub24"   ]

		doScimes.combinSubCubes(CO12FITSAll,cubeList )



		sys.exit()

	region=sys.argv[1]

	print "Region Name: ", region

	doScimes=mySCIMES( region )

	doScimes.subFITSPath="./mosaicTest1/"

	doScimes.saveSCIMESPath="./mosaicTest1Save/"


	CO12FITS=doScimes.subFITSPath+ region+".fits"
	hdu=fits.open(CO12FITS)[0]
	# maskfiles
	maskedFITS=doScimes.subFITSPath+ region+"Masked.fits"
	#if not os.path.isfile(maskedFITS ):
		#dilmasking(hdu,  saveName=maskedFITS  )

	doScimes.doDendroAndScimes(CO12FITS)

if 0: # test minpix number

	#pass
	if  len(sys.argv) < 3:
		print "Not enouth parameters, quitting..."
		sys.exit()

	region=sys.argv[1]
	minPix= int( sys.argv[ 2  ] )
	print "Region Name: ", region
	print "Minimum pix number: " , minPix





if 0:
	doScimes=mySCIMES(region)
	CO12FITS=region+".fits"
	maskedFITS=region+"MaskedData.fits"

	dendroFITS=region+"{}Dendro.fits".format(minPix)

	hdu=fits.open(CO12FITS)[0]


	if not os.path.isfile(maskedFITS):
		dilmasking(hdu,saveName=maskedFITS,th=3)
	doScimes.doDendroAndScimes( maskedFITS,saveDendro=dendroFITS, minPix=minPix )

	# Default the recursion limit


sys.setrecursionlimit(1000)
