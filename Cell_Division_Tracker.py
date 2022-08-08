# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 09:36:43 2020

@author: XYZ
Ver. 4.1
"""

#%%
import os, glob, warnings, pickle, time
import numpy as np
import re
from skimage import io, measure, filters, morphology
from scipy import ndimage, signal
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')                                               # ignore warning
print('Runnung...')

#%%
# define units
px=1
um=1
nm=1E-3*(um)

# imaging parameters             
Obj_Mag=100                                                                     # the magnificant of objective
NA=1.45                                                                         # the numerical aperture
wavelength=594*(nm)                                                             # the fluorescence emission wavelength
Relay_Mag=2.5                                                                   # relay lens
pixelsize=13*(um)                                                               # the pixel size of CCD

# digital image processing parameters
thresh_method='Customized'                                                      # the method of threshold
division_coeff=0.35                                                             # determinate if division
neighbor_dist=2*(um)                                                            # safe distance between available cells
border_dist=2*(um)                                                              # define a safe border distance
available_length=8*(um)                                                         # define an available cell length

# working directory
inputdir=r'C:\Users\xiang\Documents\XYZ\Data\Cell_Division_Tracker\Database\E. coli\SOC\BF'

#%% Initialization
print('Initializing...')
listing=glob.glob(inputdir+'\\*.tif')
nFiles=len(listing)

eff_pixelsize=pixelsize/(Obj_Mag*Relay_Mag)                                     # effective pixelsize
DL=(0.5*wavelength/NA)/eff_pixelsize                                            # diffraction limitation (in pixel)

# Create a single directory
outputdir=inputdir+'\\Analyzd'
if not os.path.exists(outputdir):
    os.mkdir(outputdir)

#%% Define functions
# select target cell with Otsu's or customized method ########################
def cell_mask(Img,method):
    if (method=='Otsu'):
        thresh=filters.threshold_otsu(Img)
        bw=(Img<thresh)
    elif (method=='Customized'):
        # rough definition of background
        bw=(Img<1.5*np.min(Img))                                                # cell area
        bw=ndimage.gaussian_filter(1.0*bw,sigma=3*DL)
        bw=(bw>0)
        dc=np.mean(Img[np.where(bw==False)])                                    # assume that background is a constant value
        
        # the definition of customized extracellular boundary
        thresh=division_coeff*dc+(1-division_coeff)*np.min(Img)
        bw=(Img<thresh)
        bw=ndimage.binary_fill_holes(bw)
        kernal=morphology.disk(np.round(DL))
        bw=morphology.binary_opening(bw,kernal)
    return bw

# remove cells near border ###################################################
def cell_filter_border(bw,R):
    R=np.round(R/eff_pixelsize)                                                 # a border distance
    L=measure.label(bw,connectivity=1)
    N=np.max(L)
    upper=np.shape(L)-np.array([R,R])                                           # upper bound for x- and y-axis
    for n in range(1,N+1):
        L_n=(L==n)
        row,col=np.where(L_n==True)
        checksum=np.sum((row<=R)+(row>=upper[0])+(col<=R)+(col>=upper[1]))      # check whether cell exist on the boundary 
        if (checksum>0):
            L[row,col]=0
    bw=(L>0)
    return bw
        
# remove over long cell and calculate endpoints of major length ##############
def cell_filter_majorL(bw,R):
    R=R/eff_pixelsize                                                           # an upper bound of cell length
    L=measure.label(bw,connectivity=1)
    N=np.max(L)
    Cs=cell_outer_contours(bw)
    pts=[]                                                                      # endpoints in major axis
    for n in range(0,N):
        Cn=Cs[n]                                                                # n-th contour
        MajorLength=0
        for i in range(0,len(Cn)):                                              # i-th point in n-th contour
            dr=Cn[i]-Cn
            dist=np.sqrt(dr[:,0]**2+dr[:,1]**2)
            j=np.where(dist==np.amax(dist))
            j=j[0][0]
            if (dist[j]>MajorLength):
                MajorLength=dist[j]
                pt1=Cn[i]
                pt2=Cn[j]
        if (MajorLength<R):
            pts.append([pt1,pt2])
        else:
            L_n=(L==n+1)
            row,col=np.where(L_n==True)
            L[row,col]=0
    bw=(L>0)
    return bw, pts

# remove intimate cells ######################################################
def cell_filter_neighbors(bw,R):
    R=R/eff_pixelsize                                                           # a distance between bacteria
    L=measure.label(bw,connectivity=1)
    N=np.max(L)
    Cs=cell_outer_contours(bw)                                                  # contours
    idxs=[]                                                                     # a removing listing of intimate cell
    for n in range(0,N):
        Cn=Cs[n]                                                                # n-th contour
        checkTF=False
        for m in range(0,N):
            if (n!=m):
                Cm=Cs[m]                                                        # m-th contour
                for i in range(0,len(Cn)):                                      # i-th point in n-th contour
                    dr=Cn[i]-Cm
                    dist=np.sqrt(dr[:,0]**2+dr[:,1]**2)
                    checksum=np.sum((dist<R))
                    if (checksum>0):
                        idxs.append(n+1)
                        checkTF=True
                        break
            if (checkTF):
                break
    for n in range(0,len(idxs)):
        L_n=(L==idxs[n])
        row,col=np.where(L_n==True)
        L[row,col]=0
    bw=(L>0)
    return bw

# calculate cell centroids ###################################################
def cell_centroids(bw):
    L=measure.label(bw,connectivity=1)
    N=np.max(L)
    pts=[np.flip(np.mean(np.where((L==n)==True),1),0) for n in range(1,N+1)]    # centroids
    return pts

# find contour ###############################################################
def cell_outer_contours(bw):
    L=measure.label(bw,connectivity=1)
    N=np.max(L)                                                               
    Cs=[np.roll(measure.find_contours((L==n),0.5)[0],1,axis=1) for n in range(1,N+1)] # contours
    Cs=smooth_contours(Cs)
    return Cs

# smooth binary contour ######################################################
def smooth_contours(Cs):
    for n in range(0,len(Cs)):
        rho,phi=cart2pol(Cs[n][:,0],Cs[n][:,1])
        T=5
        N=len(phi)
        phi_extend=np.insert(phi,0,phi[N-T:N])
        phi_extend=np.append(phi_extend,phi[0:T])
        phi_extend=np.convolve(phi_extend,np.ones((T,))/T,mode='same')
        rho_extend=np.insert(rho,0,rho[N-T:N])
        rho_extend=np.append(rho_extend,rho[0:T])
        rho_extend=np.convolve(rho_extend,np.ones((T,))/T,mode='same')
        phi=phi_extend[T:T+N]
        rho=rho_extend[T:T+N]
        Cs[n][:,0],Cs[n][:,1]=pol2cart(rho,phi)
    return Cs

# convert cartesian coordinate into polar coordinate #########################
def cart2pol(x,y):
    rho=np.sqrt(x**2+y**2)
    phi=np.arctan2(y,x)
    return rho, phi

# convert polar coordinate into cartesian coordinate #########################
def pol2cart(rho,phi):
    x=rho*np.cos(phi)
    y=rho*np.sin(phi)
    return x, y

# label number and find countour #############################################
def bwlabel(bw):
    L=measure.label(bw,connectivity=1)
    N=np.max(L)
    Cs=[]
    epts=[]
    for n in range(1,N+1):
        L_n=(L==n)
        L_n=otsu_correct(L_n)
        row,col=np.where(L_n==True)
        L[row,col]=n
        _,ept=cell_filter_majorL(L_n,np.Inf)
        epts.append(ept[0])
        C=cell_outer_contours(L_n)
        Cs.append(C[0])
    return L, N, Cs, epts

# the location of the tracked cell in the next frame #########################
def cell_WhereAmI(bw,old_L):
    L=measure.label(bw,connectivity=1)
    L_backup=L
    old_bw=(old_L>0)
    
    # shift correction
    dx,dy=fast_xcorr2(bw,old_bw)                                                # calculate 2d correlation by FFT
    old_L=np.roll(old_L,dy,axis=0)
    old_L=np.roll(old_L,dx,axis=1)
    old_bw=(old_L>0)
    
    L=L*(old_bw)
    bw0=[(L_backup==num) for num in list(set(L[np.where(L>0)]))]
    bw=(sum(bw0)>0)
    return bw
    
# create relation between current and previous frame #########################
def cell_conncetion(bw,old_L):
    print('Create cell connection list between frames...')
    L=measure.label(bw,connectivity=1)
    N=np.max(L)
    old_nums=[]
    new_nums=[]
    if (old_L.size==0):
        for n in range(1,N+1):
            L_n=(L==n)
            new_nums.append([n])
        connection_list=np.array(new_nums)
    else:
        # shift correction
        dx,dy=fast_xcorr2(bw,(old_L>0))                                         # calculate 2d correlation by FFT
        old_L=np.roll(old_L,dy,axis=0)
        old_L=np.roll(old_L,dx,axis=1)
        
        for n in range(1,N+1):
            L_n=(L==n)
            m=int(np.median(old_L[np.where(L_n==True)]))
            old_nums.append(m)
            new_nums.append(n)
        connection_list=np.array(np.transpose([old_nums,new_nums]))             # [previous, current]
    return connection_list

# convert conncetion list into a chain-code array ############################
def cell_connection_chaincode(connection_lists):
    print('Convert the conncetion list into a chain-code array...')
    nSteps=len(connection_lists)                                                # nSteps=nFrames
    nSeeds=np.shape(connection_lists[nSteps-1])[0]
    chaincode_array=np.zeros((nSeeds,nSteps), dtype=np.int)
    brokenCounts=0
    for nSeed in range(0,nSeeds):
        for nStep in range(nSteps-1,-1,-1):
            if (nStep==nSteps-1):
                chaincode_array[nSeed,nStep]=connection_lists[nStep][nSeed,1]
                previous_num=connection_lists[nStep][nSeed,0]
            elif (nStep==0):
                chaincode_array[nSeed,nStep]=previous_num
                idx=(np.where(connection_lists[nStep][:,0]==previous_num))[0]
                if (len(idx)==0):                                               # if the cell lost contact, then break loop to next cell
                    brokenCounts=brokenCounts+1    
                    break
                previous_num=connection_lists[nStep][idx,0]
            else:
                chaincode_array[nSeed,nStep]=previous_num
                idx=(np.where(connection_lists[nStep][:,1]==previous_num))[0]
                if (len(idx)==0):
                    brokenCounts=brokenCounts+1  
                    break
                previous_num=connection_lists[nStep][idx,0]
    return chaincode_array, brokenCounts
    
# feature structure for labeling map, contours, centroids, and endpoints #####
def cell_container(L,Cs,CMs,pts):
    N=np.max(L)
    container=[[Cs[n],CMs[n],pts[n]] for n in range(0,N)]
    return container
 
# two-dimensional fast cross-correlation #####################################
def fast_xcorr2(bw1,bw2):
    bw2=1*bw2
    corr=signal.fftconvolve(1*bw1,bw2[::-1,::-1],mode='same')
    y,x=np.unravel_index(np.argmax(corr), corr.shape)
    shift_y=int(y-np.shape(bw1)[0]/2)
    shift_x=int(x-np.shape(bw1)[1]/2)
    return shift_x, shift_y

# consistent boundary with Otsu's threshold
def otsu_correct(bw):  
    kernal=morphology.disk(5)
    bw=morphology.binary_dilation(bw,kernal)
    return bw

# draw elements after image process ##########################################
def drawElements(Img,Cs,CMs,epts,label_num_list):                               # Cs: contours, CMs: centroids, epts: endpoints
    print('Draw elements...')
    fig, ax = plt.subplots()
    plt.get_current_fig_manager().window.showMaximized()
    ax.imshow(Img, cmap=plt.cm.gray)   
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    N=len(CMs)
    for n in range(0,N):
        ax.plot(Cs[n][:,0],Cs[n][:,1])
        ax.plot(epts[n][0][0],epts[n][0][1],'r+')
        ax.plot(epts[n][1][0],epts[n][1][1],'r+')
        if (label_num_list==np.Inf):
            ax.text(CMs[n][0],CMs[n][1],str(n+1),color='white')
        else:
            ax.text(CMs[n][0],CMs[n][1],str(label_num_list[n]),color='white')
    plt.pause(0.05)

#%% Main
t0=time.time()
TrackLists=[]
TrackTrees=[]
for nFile in range(0,nFiles):
    # read file
    print('Load files...(current file: '+str(nFile+1)+' / total files: '+str(nFiles)+')')
    inputfile=listing[nFile]
    BF=io.imread(inputfile)                                                     # bright-field image
    nFrames=np.shape(BF)[0]
    
    # image process
    Ss=[]                                                                       # feature structure for labeling map, contours, centroids, and endpoints
    branches=[]                                                                 # collect connection list
    for nFrame in range(0,nFrames):
        print('Image processing...(current frame: '+str(nFrame+1)+' / total frames: '+str(nFrames)+')')
        origina=BF[nFrame,:,:]
        if (nFrame==0):
            mask=cell_mask(origina,thresh_method)                               # selecting target by Otsu or FWHM           
            mask=cell_filter_neighbors(mask,neighbor_dist)                      # filter out intimate cells
            mask=cell_filter_border(mask,border_dist)                           # filter out the cell near the border
            mask,_=cell_filter_majorL(mask,available_length)                    # filter out over short or long cells and calculate endpoints
            branches.append(cell_conncetion(mask,np.array([])))                 # create connetion between current and previous
            centroids=cell_centroids(mask)                                      # calculate center point of target cell
            label_map,nCells,contours,endpoints=bwlabel(mask)                   # label number and find contours
            S=cell_container(label_map,contours,centroids,endpoints)            # the container includes centroid, endpoints, and contour for each of tracking cells
            drawElements(origina,contours,centroids,endpoints,np.Inf)           # draw elements for all cell
            print('There are '+str(nCells)+' available mother cells.')
            
            # save the figure to file
            print('Save the labelled map...')
            outputfile=outputdir+'\\'+re.compile('\w+.tif').findall(listing[nFile])[0]
            plt.gcf().savefig(outputfile,bbox_inches='tight',pad_inches=0)               
            plt.close(plt.gcf())                                                # close the figure window
        else:
            mask=cell_mask(origina,thresh_method)  
            mask=cell_WhereAmI(mask,label_map)          
            mask,_=cell_filter_majorL(mask,np.Inf)
            branches.append(cell_conncetion(mask,label_map))  
            centroids=cell_centroids(mask)
            label_map,nCells,contours,endpoints=bwlabel(mask)
            S=cell_container(label_map,contours,centroids,endpoints)
            # drawElements(origina,contours,centroids,endpoints,np.Inf) 
        Ss.append(S)
    
    # collect tracking data
    Tree,lostCounts=cell_connection_chaincode(branches) 
    TrackTrees.append(Tree)                                                     # [nFile]
    TrackLists.append(Ss)                                                       # [nFile][nFrame][nCell]
    print('There are '+str(nCells-lostCounts)+' available daughter cells.') 
t1=time.time()
print('Elapsed time is '+str(t1-t0)+' seconds.')

#%% save database
print('Save analyzed datasets into the computer....')
outputfile=outputdir+'\\Datasets_BF'
Datasets_BF=[TrackTrees,TrackLists]
with open(outputfile,'wb') as f:
    pickle.dump(Datasets_BF,f)

#%%
print('Done...')

#%% Draw specified single-cell tracking process
ans=input('Do you want to check? [y/n]:')
if (ans=='y'):
    ans=input('nFile:')
    obs_nFile=int(ans)
    ans=input('nCell:')
    obs_nCell_num=int(ans)
    
    print('You choose the mother cell: '+str(obs_nCell_num)+' in the file: '+str(obs_nFile)) 
    nFile=obs_nFile-1
    inputfile=listing[nFile]
    BF=io.imread(inputfile)                                                     
    Tree=TrackTrees[nFile]
    plot_branch_idxs=(np.where(Tree[:,0]==obs_nCell_num))[0]

    for nFrame in range(0,np.shape(BF)[0]):
        origina=BF[nFrame,:,:]
        contours=[]
        centroids=[]
        endpoints=[]
        current_cell_nums=[]
        for nPlot in range(0,len(plot_branch_idxs)):
            current_cell_num=Tree[plot_branch_idxs[nPlot],nFrame]
            current_cell_nums.append(current_cell_num)
            contours.append(TrackLists[nFile][nFrame][current_cell_num-1][0])
            centroids.append(TrackLists[nFile][nFrame][current_cell_num-1][1])
            endpoints.append(TrackLists[nFile][nFrame][current_cell_num-1][2])
        drawElements(origina,contours,centroids,endpoints,current_cell_nums)
    