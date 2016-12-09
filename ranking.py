import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn import preprocessing as pre
from sklearn import metrics



data = pd.read_csv("GT_mclean.csv")


##########################
#This is a general cluster helper function using DBSCAN algorithm
def cluster(mask,ESP,minN):
    d = pd.read_csv("GT_mclean.csv")
    d.loc[~mask,'latitude'] = 0.0
    d.loc[~mask,'longitude'] = 0.0
    lat = d['latitude']
    lat = lat[pd.notnull(lat)]  #eliminate nan value
    lon = d['longitude']
    lon = lon[pd.notnull(lon)]
    lat_lon = pd.concat([lat,lon],axis=1)

    lat_lon = lat_lon.as_matrix().astype("float32",copy=False)
    #stscaler = pre.StandardScaler().fit(lat_lon)
    #lat_lon = stscaler.transform(lat_lon)

    dbsc = DBSCAN(eps=ESP, min_samples = minN).fit(lat_lon)

    labels = dbsc.labels_
    


    ##########################
    # codes below are for plotting
    ##########################

    '''
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
    plt.show()

    '''

 

    if len(labels) < len(data):
        for i in range(len(data) - len(labels)):
            labels = np.append(labels,-1)
   

    return labels





    
#classification based on density
def classifyByLetter():
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
    

#######################
# In function classifyByScore(), instead of putting each location point into
# the letter classes, it gives scores of seriousness to points
#(A-0,B-2,C-4,D-6,E-8,F-10) and return the score list.
#######################
def classifyByScore():
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

    

    labels0 = dbsc0.labels_
    labels1 = dbsc1.labels_
    labels2 = dbsc2.labels_
    labels3 = dbsc3.labels_
    labels4 = dbsc4.labels_
    labels5 = dbsc5.labels_

    rank = [0]*len(data)

    mask0 = (labels0 != -1)
    
    mask1 = (labels1 != -1)
    mask2 = (labels2 != -1)
    mask3 = (labels3 != -1)
    mask4 = (labels4 != -1)
    mask5 = (labels5 == -1)
    
    for i in range(len(labels0)):
        if lat_lon[i] in lat_lon[mask0]:
            rank[i] = 10
        elif lat_lon[i] in lat_lon[mask1]:
            rank[i] = 8
        elif lat_lon[i] in lat_lon[mask2]:
            rank[i] = 6
        elif lat_lon[i] in lat_lon[mask3]:
            rank[i] = 4
        elif lat_lon[i] in lat_lon[mask4]:
            rank[i] = 2
        else:
            rank[i] = 0
        

    #if GT_with_Rank_score.csv file has not been generated yet, uncomment
    #the code below to output the result file
    '''
    d = data
    d['density_score'] = pd.Series(rank)
    df = pd.DataFrame(d)
    #df.to_csv('GT_with_Rank_score.csv')
    df.to_csv('GT_modified.csv')
    '''
    # return type: list
    return rank



#classification based on seasonal distribution
#take a file to write results in, GT_modified.csv set as default file
def seasonalClass(filename='GT_modified.csv'):
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

        
    spmask = (cluster(spring,0.001,30) != -1)
    sumask = (cluster(summer,0.001,30) != -1)
    famask = (cluster(fall,0.001,30) != -1)
    wtmask = (cluster(winter,0.001,30) != -1)

    # weighted scores of seriousness for each season: spring and fall 7,
    # summer 6,winter 4

    for i in range(len(data)):
        if (spmask[i] & spring[i]) == True:
            seasonclass[i] = '8'
        if (sumask[i] & summer[i]) == True:
            seasonclass[i] = '6'
        if (famask[i] & fall[i]) == True:
            seasonclass[i] = '7'
        if (wtmask[i] & winter[i]) == True:
            seasonclass[i] = '4'

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

    '''    
    for k in hashTb:
        print k,':',hashTb[k]
    '''

    # turn string into int
    
    for i in hashTb:
        nSum = 0
        for j in range(len(hashTb[i])):
            nSum += int(hashTb[i][j])
        hashTb[i] = nSum
            
    seasonScore = [0]*len(data)      
    # uniform the scores for locations with same latitude and longitude
    # updated scores stored in list seasonclass by index of locations
    for i in range(len(data)):
        row = data.loc[i]
        lat = row['latitude']
        lon = row['longitude']
        loc = str(lat)+","+str(lon)

        if loc in hashTb:
            seasonScore[i] = hashTb[loc]

    # uncomment to write in modifed file
    '''
    frame = pd.read_csv(filename)
    frame['seasonal_score'] = pd.Series(seasonScore)
    df = pd.DataFrame(frame)
    #generate separated file
    #df.to_csv('GT_with_Rank_and_Season.csv')

    #write in same file
    df.to_csv('GT_modified.csv')
    '''

    return seasonScore


#helper method to get score for each type of crime
def getTypeScore(filename = 'GT_modified.csv'):
    frame = pd.read_csv(filename)
    typeScoreList = [0]*len(frame)
    for i in range(len(frame)):
        row = frame.loc[i]
        cType = row['parent_incident_type']
        if (cType == 'Sexual Assault') or (cType == 'Robbery'):
            typeScoreList[i] = 12
        elif (cType == 'Kidnapping') or (cType == 'Assault with Deadly Weapon')\
             or (cType == 'Other Sexual Offense') or (cType == 'Theft of Vehicle')\
             or (cType == 'Arson'):
            typeScoreList[i] = 10
        elif (cType == 'Breaking & Entering') or (cType == 'Drugs')\
             or (cType == 'Theft') or (cType == 'Traffic'):
            typeScoreList[i] = 8
        elif (cType == 'Missing Person') or (cType == 'Assault')\
             or (cType == 'Theft from Vehicle'):
            typeScoreList[i] = 6
        elif (cType == 'Disorder') or (cType == 'Liquor')\
             or (cType == 'Weapons Offense') or (cType == 'Property Crime'):
            typeScoreList[i] = 4
        elif (cType == 'Other') or (cType == 'Family Offense'):
            typeScoreList[i] = 2
    '''
    frame['type_score'] = pd.Series(typeScoreList)
    df = pd.DataFrame(frame)
    df.to_csv('GT_modified.csv')
    '''
    
    return typeScoreList


def avgTypeScore(filename='GT_modified.csv'):
    typeScoreList = np.asarray(getTypeScore(),dtype='float')
    avgSList = typeScoreList
    ratioList = [0.0]*len(data)
    mask = np.asarray([True]*(len(data)))
    labels = cluster(mask,0.0006,10)
    

    unique_labels = set(labels)

    for k in unique_labels:
        if(k != -1):
            class_member_mask = (labels == k)
            nPoints = sum(class_member_mask == True)
            scoreSubList = typeScoreList[class_member_mask]
            avg = sum(scoreSubList)*1.0/nPoints
            avgSList[class_member_mask] = avg
        

    
    frame = pd.read_csv(filename)
    frame['type_avg_score'] = pd.Series(avgSList)
    '''
    typeScoreList = list(typeScoreList)
    avgSList = list(avgSList)

    for i in range(len(data)):
        ratioList[i] = typeScoreList[i]*1.0/float(avgSList[i])
        
  
    frame['type_avg_ratio'] = pd.Series(ratioList)
    '''
    df = pd.DataFrame(frame)
    df.to_csv('GT_modified.csv')
   
    return ratioList
#avgTypeScore()

#does not work,abandon
''' 
from sklearn import tree        
def classTree(filename='GT_modified.csv'):
    frame = pd.read_csv(filename)
    features = ['latitude','longitude','seasonal_score','type_avg_score']

    y = frame['density_score']
    x = frame[features]
    dt = DecisionTreeClassifier()
    dt = dt.fit(x,y)

    return dt
'''

#takes forever,bad idea
'''
from haversine import haversine as hrs
def getclosetpoint(lat,lon):
    frame = pd.read_csv('GT_modified.csv')
    index = 0
    mindis = hrs((frame.loc[0]['latitude'],frame.loc[0]['longitude']),(lat,lon))
    for i in range(len(data)):
        la = frame.loc[i]['latitude']
        lo = frame.loc[i]['longitude']
        loc = (la,lo)
        dis = hrs(loc,(lat,lon))
        if dis < mindis:
            mindis = dis
            index = i
    return {'seasonal_score':frame.loc[i]['seasonal_score'],'type_avg_score':frame.loc[i]['type_avg_score']}
'''        

def getRank(lat,lon):
    dic = getclosetpoint(lat,lon)
    seasonS = dic['seasonal_score']
    tAvg = dic['type_avg_score']
    dt = classTree()
    result = dt.predict(pd.Series([lat,lon,seasonS,tAvg]))
    print(result)
    

def linearRank(filename='GT_modified.csv'):
    frame = pd.read_csv(filename)
    rank = ['']*len(data)
    for i in range(len(frame)):
        row = frame.loc[i]
        tAvgScore = row['type_avg_score']
        sScore = row['seasonal_score']
        dScore = row['density_score']
        score = 0.5*dScore+0.3*tAvgScore+0.2*sScore
        if score <= 3:
            rank[i] = 'A'
        elif score <= 5:
            rank[i] = 'B'
        elif score <= 7:
            rank[i] = 'C'
        elif score <= 9:
            rank[i] = 'D'
        elif score <= 11:
            rank[i] = 'E'
        else:
            rank[i] = 'F'
    frame['rank'] = pd.Series(rank)
    df = pd.DataFrame(frame)
    df.to_csv('GT_modified.csv')

#linearRank()


def drawRank(filename='GT_modified.csv'):
    frame = pd.read_csv(filename)
    
    dColor = ''
    for i in range(len(frame)):
        row = frame.loc[i]
        dRank = row['rank']
        lat = row['latitude']
        lon = row['longitude']
        if dRank == 'A':
            dColor = '#ccffcc'
        elif dRank == 'B':
            dColor = '#00ff99'
        elif dRank == 'C':
            dColor = '#99ff33'
        elif dRank == 'D':
            dColor = '#ffff00'
        elif dRank == 'E':
            dColor = '#ff9933'
        else:
            dColor = '#ff3300'
        plt.plot(lon,lat,'o',markerfacecolor=dColor)

    
    plt.show()
#drawRank()



