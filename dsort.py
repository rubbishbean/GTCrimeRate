import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("GT_mclean.csv")

def sortSeason():
    #data = pd.read_csv("GT_mclean.csv")
    #print(data.columns)
    dateTime = data['incident_datetime']
    springList = []
    summerList = []
    fallList = []
    winterList = []
    #dSeasons = {'Spring':pd.Series([]),'Summer':pd.Series([]),'Fall':pd.Series([]),'Winter':pd.Series([])}
    
    for row in dateTime:
        if int(row.split('/')[0]) in range(3,6):
            springList.append(row)
        elif int(row.split('/')[0]) in range(6,9):
            summerList.append(row)
        elif int(row.split('/')[0]) in range(9,12):
            fallList.append(row)
        elif int(row.split('/')[0]) in [12,1,2]:
            winterList.append(row)

    dSeasons = {'Spring':pd.Series(springList),'Summer':pd.Series(summerList),'Fall':pd.Series(fallList),'Winter':pd.Series(winterList)}

    df = pd.DataFrame(dSeasons)
    #print(df.columns)
    df.to_csv('seasonSort.csv')


def sortAtSeason():
    #data = pd.read_csv("GT_mclean.csv")
    #print(data.columns)
    dateTime = data['incident_datetime']
    springList = []
    summerList = []
    fallList = []
    winterList = []
    #dSeasons = {'Spring':pd.Series([]),'Summer':pd.Series([]),'Fall':pd.Series([]),'Winter':pd.Series([])}
    
    for row in dateTime:
        if int(row.split('/')[0]) in range(3,6):
            springList.append(row)
        elif int(row.split('/')[0]) in range(6,9):
            summerList.append(row)
        elif int(row.split('/')[0]) in range(9,13):
            fallList.append(row)
        elif int(row.split('/')[0]) in range(1,3):
            winterList.append(row)

    dSeasons = {'Spring':pd.Series(springList),'Summer':pd.Series(summerList),'Fall':pd.Series(fallList),'Winter':pd.Series(winterList)}

    df = pd.DataFrame(dSeasons)
    #print(df.columns)
    df.to_csv('seasonSort_at.csv')

def sortByYear():
    
    #dateTime = data['incident_datetime']
    listOfYears = [[] for i in range(11)]
    #print(data.loc[1]['incident_datetime'])
    #print(len(data))

    for i in range(len(data)):
        row = data.loc[i]
        
        index = int(row['incident_datetime'].split(' ')[0].split("/")[2]) - 2006
        listOfYears[index].append(row)

    dYears = {'2006':pd.Series(listOfYears[0]), '2007':pd.Series(listOfYears[1]), \
              '2008':pd.Series(listOfYears[2]), '2009':pd.Series(listOfYears[3]), \
              '2010':pd.Series(listOfYears[4]), '2011':pd.Series(listOfYears[5]), \
              '2012':pd.Series(listOfYears[6]), '2013':pd.Series(listOfYears[7]), \
              '2014':pd.Series(listOfYears[8]), '2015':pd.Series(listOfYears[9]), \
              '2016':pd.Series(listOfYears[10]),}
    df = pd.DataFrame(dYears)
    df.to_csv('SortByYear.csv')


def locByYear():
    listOfYears = [[] for i in range(11)]
    
    for i in range(len(data)):
        row = data.loc[i]
        
        index = int(row['incident_datetime'].split(' ')[0].split("/")[2]) - 2006
        lat = row['latitude']
        lon = row['longitude']
        loc = str(lat)+","+str(lon)
        listOfYears[index].append(loc)
        
    dYears = {'2006':pd.Series(listOfYears[0]), '2007':pd.Series(listOfYears[1]), \
              '2008':pd.Series(listOfYears[2]), '2009':pd.Series(listOfYears[3]), \
              '2010':pd.Series(listOfYears[4]), '2011':pd.Series(listOfYears[5]), \
              '2012':pd.Series(listOfYears[6]), '2013':pd.Series(listOfYears[7]), \
              '2014':pd.Series(listOfYears[8]), '2015':pd.Series(listOfYears[9]), \
              '2016':pd.Series(listOfYears[10]),}
    df = pd.DataFrame(dYears)
    df.to_csv('LocByYear.csv')

def drawLocByYear():
    raw_data = pd.read_csv("LocByYear.csv")
    listOfYears = [[] for i in range(11)]

    for i in range(11):
        column = str(i+2006)
        listOfYears[i] = raw_data[column]

    listOfLat = [[] for i in range(11)]
    listOfLon = [[] for i in range(11)]
    
    for i in range(11):
        currentY = listOfYears[i]
        currentY = currentY[pd.notnull(currentY)]  #####get rid of NAN!!!!!!#####
        for row in currentY:
            #print((str(row).split(','))[0])
            lat = (str(row).split(','))[0]
            #print(lat+ " " + row)
            lon = (str(row).split(','))[1]

            listOfLat[i].append(float(lat))
            listOfLon[i].append(float(lon))

    lat_max = max([max(sublat) for sublat in listOfLat])
    lat_min = min([min(sublat) for sublat in listOfLat])
    lon_max = max([max(sublat) for sublat in listOfLon])
    lon_min = min([min(sublat) for sublat in listOfLon])

    X,Y=np.mgrid[lat_min:lat_max:100j,lon_min:lon_max:100j]
    plt.plot(listOfLon[0],listOfLat[0],'ro')
    #plt.plot(listOfLon[1],listOfLat[1],'bo')
    #plt.plot(listOfLon[2],listOfLat[2],'go')
    #plt.plot(listOfLon[3],listOfLat[3],'ko')
    #plt.plot(listOfLon[4],listOfLat[4],'yo')
    plt.plot(listOfLon[9],listOfLat[9],'co')
    
    plt.show()



def drawBySeason():
    #data = pd.read_csv("GT_mclean.csv")
    #print(data.columns)
    springLoc = []
    summerLoc = []
    fallLoc = []
    winterLoc = []

    listOfLat = [[] for i in range(4)]
    listOfLon = [[] for i in range(4)]
    
    
    for i in range(len(data)):
        row = data.loc[i]
        lat = row['latitude']
        lon = row['longitude']
        loc = str(lat)+","+str(lon)  #for storing purpose
        if int(row['incident_datetime'].split('/')[0]) in range(3,6):
            springLoc.append(loc)
            listOfLat[0].append(float(lat))
            listOfLon[0].append(float(lon))
        elif int(row['incident_datetime'].split('/')[0]) in range(6,9):
            summerLoc.append(loc)
            listOfLat[1].append(float(lat))
            listOfLon[1].append(float(lon))
        elif int(row['incident_datetime'].split('/')[0]) in range(9,13):
            fallLoc.append(loc)
            listOfLat[2].append(float(lat))
            listOfLon[2].append(float(lon))
        elif int(row['incident_datetime'].split('/')[0]) in range(1,3):
            winterLoc.append(loc)
            listOfLat[3].append(float(lat))
            listOfLon[3].append(float(lon))

   # dSeasons = {'Spring':pd.Series(springLoc),'Summer':pd.Series(summerLoc),\
               # 'Fall':pd.Series(fallLoc),'Winter':pd.Series(winterLoc)}

    #df = pd.DataFrame(dSeasons)
    #print(df.columns)
    #df.to_csv('LocBySeasons.csv')

    plt.figure(1)
    plt.subplot(221)
    plt.plot(listOfLon[0],listOfLat[0],'ro')

    plt.subplot(222)
    plt.plot(listOfLon[1],listOfLat[1],'bo')

    plt.subplot(223)
    plt.plot(listOfLon[2],listOfLat[2],'co')

    plt.subplot(224)
    plt.plot(listOfLon[3],listOfLat[3],'go')
    #plt.grid(True)
    plt.show()


'''
Type number: 0-Assault, 1-Assault with Deadly Weapon, 2-Breaking & Entering,
3-Disorder, 4-Drugs, 5-Liquor, 6-Other(potetial), 7-Other Sexual Offense,
8-Sexual Assault, 9-Property Crime(fraud&public), 10-Robbery(strong-armed),
11-Theft(bike&building&pocket), 12-Theft from Vehicle, 13-Theft of Vehicle,
14-Traffic
'''

def sortByType():
    listOfTypes = [[] for i in range(15)]
    listOfLat = [[] for i in range(15)]
    listOfLon = [[] for i in range(15)]
    for i in range(len(data)):
        row = data.loc[i]
        lat = row['latitude']
        lon = row['longitude']
        loc = str(lat)+","+str(lon)  #for storing purpose
        if(row['parent_incident_type'] == 'Assault'):
            listOfTypes[0].append(loc)
            listOfLat[0].append(float(lat))
            listOfLon[0].append(float(lon))
        elif (row['parent_incident_type'] == 'Assault with Deadly Weapon'):
            listOfTypes[1].append(loc)
            listOfLat[1].append(float(lat))
            listOfLon[1].append(float(lon))
        elif (row['parent_incident_type'] == 'Breaking & Entering'):
            listOfTypes[2].append(loc)
            listOfLat[2].append(float(lat))
            listOfLon[2].append(float(lon))
        elif (row['parent_incident_type'] == 'Disorder'):
            listOfTypes[3].append(loc)
            listOfLat[3].append(float(lat))
            listOfLon[3].append(float(lon))
        elif (row['parent_incident_type'] == 'Drugs'):
            listOfTypes[4].append(loc)
            listOfLat[4].append(float(lat))
            listOfLon[4].append(float(lon))
        elif (row['parent_incident_type'] == 'Liquor'):
            listOfTypes[5].append(loc)
            listOfLat[5].append(float(lat))
            listOfLon[5].append(float(lon))
        elif (row['parent_incident_type'] == 'Other'):
            listOfTypes[6].append(loc)
            listOfLat[6].append(float(lat))
            listOfLon[6].append(float(lon))
        elif (row['parent_incident_type'] == 'Other Sexual Offense'):
            listOfTypes[7].append(loc)
            listOfLat[7].append(float(lat))
            listOfLon[7].append(float(lon))
        elif (row['parent_incident_type'] == 'Sexual Assault'):
            listOfTypes[8].append(loc)
            listOfLat[8].append(float(lat))
            listOfLon[8].append(float(lon))
        elif (row['parent_incident_type'] == 'Property Crime'):
            listOfTypes[9].append(loc)
            listOfLat[9].append(float(lat))
            listOfLon[9].append(float(lon))
        elif (row['parent_incident_type'] == 'Robbery'):
            listOfTypes[10].append(loc)
            listOfLat[10].append(float(lat))
            listOfLon[10].append(float(lon))
        elif (row['parent_incident_type'] == 'Theft'):
            listOfTypes[11].append(loc)
            listOfLat[11].append(float(lat))
            listOfLon[11].append(float(lon))
        elif (row['parent_incident_type'] == 'Theft from Vehicle'):
            listOfTypes[12].append(loc)
            listOfLat[12].append(float(lat))
            listOfLon[12].append(float(lon))
        elif (row['parent_incident_type'] == 'Theft of Vehicle'):
            listOfTypes[13].append(loc)
            listOfLat[13].append(float(lat))
            listOfLon[13].append(float(lon))
        elif (row['parent_incident_type'] == 'Traffic'):
            listOfTypes[14].append(loc)
            listOfLat[14].append(float(lat))
            listOfLon[14].append(float(lon))

    #plt.plot(listOfLon[2],listOfLat[2],'ro')
    #plt.plot(listOfLon[10],listOfLat[10],'b*')
    plt.plot(listOfLon[7],listOfLat[7],'gs')
    plt.plot(listOfLon[4],listOfLat[4],'r*')
    
    plt.show()
    
    
     
        
#sortByYear()
#sortAtSeason()    
#locByYear()    
#drawLocByYear()
#drawBySeason()

#sortByType()


from sklearn.cluster import DBSCAN
from sklearn import preprocessing as pre
from sklearn import metrics

###############################################################
def clusterAll():
    lat = data['latitude']
    lat = lat[pd.notnull(lat)]
    lon = data['longitude']
    lon = lon[pd.notnull(lon)]
    lat_lon = pd.concat([lat,lon],axis=1)
    #print(lat_lon.isnull().any().any())
    #print(lat_lon)

    lat_lon = lat_lon.as_matrix().astype("float32",copy=False)
    stscaler = pre.StandardScaler().fit(lat_lon)
    lat_lon = stscaler.transform(lat_lon)
    #print(lat_lon)  #type = numpy.ndarray

    #plt.plot([x[1] for x in lat_lon], [x[0] for x in lat_lon], 'ro')
    #plt.show()
    #eps: radius
    dbsc = DBSCAN(eps=.0015, min_samples = 1400).fit(lat_lon)
    #print(type(dbsc))
    labels = dbsc.labels_
    #print(type(labels))
    core_samples = np.zeros_like(labels,dtype = bool)
    core_samples[dbsc.core_sample_indices_] = True

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)

    unique_labels = set(labels)
    #print(unique_labels)
    #print(len(unique_labels))
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -1:
        # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)
        #print(class_member_mask )

        xy = lat_lon[class_member_mask & core_samples]
        plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col,\
             markeredgecolor='k', markersize=8)

        xy = lat_lon[class_member_mask & ~core_samples]
        plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col,\
             markeredgecolor='k', markersize=3)
    plt.title('Estimated number of clusters: %d, esp=0.0015' % n_clusters_)
    plt.show()


   
def classify():
    lat = data['latitude']
    lat = lat[pd.notnull(lat)]
    lon = data['longitude']
    lon = lon[pd.notnull(lon)]
    lat_lon = pd.concat([lat,lon],axis=1)

        

    lat_lon = lat_lon.as_matrix().astype("float32",copy=False)
    stscaler = pre.StandardScaler().fit(lat_lon)
    lat_lon = stscaler.transform(lat_lon)

    # not -1 : F, cluster-num=1
    dbsc0 = DBSCAN(eps=.0015, min_samples = 1400).fit(lat_lon)
    # not -1 & not F : E, cluster-num=3
    dbsc1 = DBSCAN(eps=.0015, min_samples = 900).fit(lat_lon)
    # not -1 & not F & not E :D, cluster-num=5
    dbsc2 = DBSCAN(eps=.0015, min_samples = 600).fit(lat_lon)
    # not -1 & not F & not E & not D: C, cluster-num=5
    dbsc3 = DBSCAN(eps=.0015, min_samples = 250).fit(lat_lon)
    # not -1 & not F & not E & not D & not C: B, cluster-num=3
    dbsc4 = DBSCAN(eps=.0015, min_samples = 80).fit(lat_lon)
    #-1：A,cluster-num=7
    dbsc5 = DBSCAN(eps=.0015, min_samples = 20).fit(lat_lon)

    new_latlon = []
    rank = []

    labels0 = dbsc0.labels_
    labels1 = dbsc1.labels_
    labels2 = dbsc2.labels_
    labels3 = dbsc3.labels_
    labels4 = dbsc4.labels_
    labels5 = dbsc5.labels_

    mask0 = (labels0 != -1)
    
    mask1 = (labels1 != -1)
    mask2 = (labels2 != -1)
    mask3 = (labels3 != -1)
    mask4 = (labels4 != -1)
    mask5 = (labels5 == -1)
    for i in lat_lon[mask0]:
        new_latlon.append(i)
        rank += 'F'

    for i in lat_lon[mask1 & ~mask0]:
        new_latlon.append(i)
        rank += 'E'

    for i in lat_lon[mask2 & ~mask1 & ~mask0]:
        new_latlon.append(i)
        rank += 'D'

    for i in lat_lon[mask3 & ~mask2 & ~mask1 & ~mask0]:
        new_latlon.append(i)
        rank += 'C'

    for i in lat_lon[mask4 & ~mask3 & ~mask2 & ~mask1 & ~mask0]:
        new_latlon.append(i)
        rank += 'B'

    for i in lat_lon[mask5 & ~mask4 & ~mask3 & ~mask2 & ~mask1 & ~mask0]:
        new_latlon.append(i)
        rank += 'A'

           
    
       
    dClass = {'Lat_Lon':pd.Series(new_latlon),'Rank':pd.Series(rank)}
    df = pd.DataFrame(dClass)
    df.to_csv('Rank.csv')

def cluster(mask):
    d = pd.read_csv("GT_mclean.csv")
    d.loc[~mask,'latitude'] = 0.0
    d.loc[~mask,'longitude'] = 0.0
    lat = d['latitude']
    lat = lat[pd.notnull(lat)]
    #print(lat)
    lon = d['longitude']
    lon = lon[pd.notnull(lon)]
    #print(lon)
    lat_lon = pd.concat([lat,lon],axis=1)

    lat_lon = lat_lon.as_matrix().astype("float32",copy=False)
    #stscaler = pre.StandardScaler().fit(lat_lon)
    #lat_lon = stscaler.transform(lat_lon)

    dbsc = DBSCAN(eps=.001, min_samples = 30).fit(lat_lon)

    labels = dbsc.labels_
    


    ##########################
    # codes below are for plotting
    ##########################
 
    core_samples = np.zeros_like(labels,dtype = bool)
    core_samples[dbsc.core_sample_indices_] = True

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)

    unique_labels = set(labels)

    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -1:
        # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = lat_lon[class_member_mask & core_samples]
        plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col,\
             markeredgecolor='k', markersize=8)

        xy = lat_lon[class_member_mask & ~core_samples]
        plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col,\
             markeredgecolor='k', markersize=3)
    plt.title('Estimated number of clusters: %d, esp=0.001' % n_clusters_)
    #plt.show()

    if len(labels) < len(data):
        labels = np.append(labels,-1)
    

    return labels
    



def seasonalClass():
    spring = np.asarray([False]*(len(data)))
    summer = np.asarray([False]*(len(data)))
    fall = np.asarray([False]*(len(data)))
    winter = np.asarray([False]*(len(data)))
    

    seasonclass = np.asarray([""]*(len(data)), dtype="a16") # string of at most length 16

    
    for i in range(len(data)):
        row = data.loc[i]
        
        if int(row['incident_datetime'].split('/')[0]) in range(3,6):
            spring[i] = True
        elif int(row['incident_datetime'].split('/')[0]) in range(6,9):
            summer[i] = True
            
        elif int(row['incident_datetime'].split('/')[0]) in range(9,13):
            fall[i] = True
        elif int(row['incident_datetime'].split('/')[0]) in range(1,3):
            winter[i] = True

        
    spmask = (cluster(spring) != -1)
    sumask = (cluster(summer) != -1)
    famask = (cluster(fall) != -1)
    wtmask = (cluster(winter) != -1)

    

    for i in range(len(data)):
        if (spmask[i] & spring[i]) == True:
            seasonclass[i] += "s1"
        if (sumask[i] & summer[i]) == True:
            seasonclass[i] += "s2"
        if (famask[i] & fall[i]) == True:
            seasonclass[i] += "s3"
        if (wtmask[i] & winter[i]) == True:
            seasonclass[i] += "s4"

    hashTb = {}

    for i in range(len(data)):
        if seasonclass[i] != "":
            row = data.loc[i]
            lat = row['latitude']
            lon = row['longitude']
            loc = str(lat)+","+str(lon)
            if(not loc in hashTb):
                hashTb[loc] = seasonclass[i]
            elif (loc in hashTb) and (not seasonclass[i] in hashTb[loc]):
                hashTb[loc] += seasonclass[i]
               
    #print the dictionary for testing
    #print seasonclass

         
    for k in hashTb:
        if len(hashTb[k]) == 8:
            
            print k,':',hashTb[k]
    


def getTypeScore():
    frame = pd.read_csv("GT_lat_lon.csv")
    typeScoreList = [0]*len(frame)
    for i in range(len(frame)):
        row = frame.loc[i]
        cType = row['type']
        if (cType == 'Sexual Assault') or (cType == 'Robbery'):
            typeScoreList[i] = 6
        elif (cType == 'Kidnapping') or (cType == 'Assault with Deadly Weapon')\
             or (cType == 'Other Sexual Offense') or (cType == 'Theft of Vehicle')\
             or (cType == 'Arson'):
            typeScoreList[i] = 5
        elif (cType == 'Breaking & Entering') or (cType == 'Drugs')\
             or (cType == 'Theft') or (cType == 'Traffic'):
            typeScoreList[i] = 4
        elif (cType == 'Missing Person') or (cType == 'Assault')\
             or (cType == 'Theft from Vehicle'):
            typeScoreList[i] = 3
        elif (cType == 'Disorder') or (cType == 'Liquor')\
             or (cType == 'Weapons Offense') or (cType == 'Property Crime'):
            typeScoreList[i] = 2
        elif (cType == 'Other') or (cType == 'Family Offense'):
            typeScoreList[i] = 1
    frame['type_score'] = pd.Series(typeScoreList)
    df = pd.DataFrame(frame)
    df.to_csv('GT_crime_score.csv')
    


from haversine import haversine as hrs
def avgTypeScore():
    frame = pd.read_csv("GT_crime_score.csv")
    avgScoreList = [0]*len(data)
    for i in range(len(frame)):
        row = frame.loc[i]
        lat = row['latitude']
        lon = row['longitude']
        typeScore = row['type_score']
        currentloc = (lat,lon)
        totalScore = typeScore
        nNeighbor = 1
        for j in range(i+1, len(frame)):
            r = frame.loc[j]
            la = r['latitude']
            lo = r['longitude']
            loc = (la,lo)
            score = r['type_score']
            if hrs(currentloc,loc) <= 0.15:
                totalScore += score
                nNeighbor += 1
                
        avgScore = totalScore*1.0/nNeighbor
        avgScoreList[i] = avgScore
 
        
   
    


#clusterAll()
#classify()
#seasonalClass()
#getTypeScore()
seasonalClass()
