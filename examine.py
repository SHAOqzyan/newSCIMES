import numpy as np
from myPYTHON import myFITS
from astropy.table import Table
doFITS=myFITS()
from myTree import dendroTree


areaCol="area_exact"


def selectTBByArea(TB,minimumArea=0.015):

	processTB=TB.copy()


	print "Total number of clouds?",len(processTB)
	processTB[areaCol] = processTB[areaCol]/3600./3600. # convert to degress

	returnTB=processTB[processTB[areaCol] >minimumArea  ]

	print len(returnTB ),"Clusters  larger than {} square degress ".format(minimumArea)





for i in np.arange(100,1600,100):

	print  "examing...",i

	testCat="sub21{}ClusterCat.fit".format(i)

	a1000=Table.read(testCat)

	selectTBByArea(a1000)


	#print   a1000