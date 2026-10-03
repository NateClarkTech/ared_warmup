#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
from itertools import combinations

import numpy as np
from scipy.spatial import distance
from bidict import bidict
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import copy
from sklearn.metrics.cluster import adjusted_rand_score
from abc import ABC, abstractmethod


class Partition:
    
    # NB: the partition works in integer labels, mostly - it expects strings from the oracle, and assigns integers
    # to them in the order they occur (NOTE: this means an MNIST digit may not get assigned an integer that matches
    # the image!
    
    # this holds all the clusters in a partition (so pairwise disjoint and includes all elements)
    def __init__(self,clusterer): 
        self.clusterer = clusterer # private?
        self.cl_list = [] # private?
        self.label_key_ = bidict() #this maps string labels to integer values
        self.next_integer_label = 0 #this value is the next integer that will be assigned for the next input string
        #we increment next_label by one everytime a new label is found
        self.all_pts_queried_FLAG = False
        
        self.num_queries = 0

        
        # data (ie X) is _not included as an attribute because of the size increase it causes (of the partition)
        # also labels (ie y) are not included because they would then need to be constantly updated...
        
        # NOTE: this is a skeleton, but doesn't really have meaning until the run() method is called
            
    def get_integer_label(self, string_label):
        #return integer label calling update_label_key if new label
        #if you do not pass a string, it will be immediately converted to a string
        #this is so that when mapping back and forth from label_key_ it is clear which direction you are mapping
        #params:
            #str string_label: the user label to be paired with a machine label
        #returns int: the machine label
        if string_label in self.label_key_.keys():
            return self.label_key_[string_label]
        return self.update_label_key(string_label)
    
    def get_string_label(self,integer_label):
        return self.label_key_.inv[integer_label]    
    
    def update_label_key(self,string_label):
        #params:
            #str string_label: the user label to be paired with an integer label
        #returns int: the integer label
        #Postconditions:
            #self.label_key[string_label] has been updated to hold a unique integer label, and next_label is advanced by 1
        self.label_key_[string_label] = self.next_integer_label
        self.next_integer_label += 1
        return self.label_key_[string_label]
            
    def create_first_cluster(self, X, oracle):
        # make "first_cluster" by...
        # select a random point (rp)
        rp_ind = random.randint(0,len(X)-1)
        # create a cluster that holds all the pts as o-pts except rp, which is the l-pt
        # just use fake label for now - we'll replace it in a minute
        fake_label = 0
        o_pt_Mis = list(range(0,rp_ind))+list(range(rp_ind+1,len(X))) # everything except random pt
        first_cluster = Cluster(X, fake_label,[rp_ind],o_pt_Mis) # now we have far point
        # query (get label for f_pt)
        self.num_queries += 1
        string_label = oracle.query(X, first_cluster.f_pt_Mi)
        integer_label = self.update_label_key(string_label)
        # return rp to o_pts
        first_cluster.o_pts_Mi.append(rp_ind)
        first_cluster.l_pts_Mi.remove(rp_ind)
        # make f_pt _the_ labeled pt for the start cluster
        # re-update f_pt, f_pt_dist in this cluster
        first_cluster.label = integer_label
        first_cluster.incorporate_far_point(X)
                
        # add this cluster to list of clusters in partition
        self.cl_list = [first_cluster]            

        if string_label == 'quit':
            return 'quit'
        else:
            return 'enter while'
        
    def process_query(self, X, query_cl_ind,f_pt_label):
        # used after getting label from query
        # use label to either incorporate or split
        if self.cl_list[query_cl_ind].label == f_pt_label: # incorporate
            self.cl_list[query_cl_ind].incorporate_far_point(X)
        else: # split
            fp_cluster, non_fp_cluster = self.cl_list[query_cl_ind].split(X, f_pt_label, self.clusterer)
            # delete the split cluster
            del(self.cl_list[query_cl_ind])
            # insert the 2 new clusters into the cluster list
            self.cl_list.append(fp_cluster)
            self.cl_list.append(non_fp_cluster)
                
    def select_query_cluster_ind(self):
        dists = []
        for x in self.cl_list:
            dists.append(x.f_pt_dist)
        max_dist = np.array(dists).max()
        if max_dist == 0: # we've queried every point
            max_ind = -1
        else:
            max_ind = np.array(dists).argmax()
        return max_ind

        
    def predict_labels(self, X):
    # goes through clusters and
    # returns:
    #    labels - 1-d array of predicted labels
        l_pts_labels = -1*np.ones((X.shape[0]), dtype=int)
        o_pts_labels = -1*np.ones((X.shape[0]), dtype=int)
        all_pts_labels = -1*np.ones((X.shape[0]), dtype=int)
        for cl in self.cl_list:
            this_label = cl.label
            l_pts_labels[cl.l_pts_Mi] = this_label
            o_pts_labels[cl.o_pts_Mi] = this_label
            all_pts_labels[cl.l_pts_Mi + cl.o_pts_Mi] = this_label 
        return all_pts_labels,l_pts_labels, o_pts_labels
            
    def run(self, X, oracle):
        
        
        #initialize the first cluster if it does not exist
        if not self.cl_list:
            string_f_pt_label = self.create_first_cluster(X,oracle)
            
        #Do
        string_f_pt_label = "go" # just a value != 'quit'
        #While
        while string_f_pt_label != "quit":
            
            q_cl_ind = self.select_query_cluster_ind()
            if q_cl_ind == -1: # got all the way to every point being queried
                self.all_pts_queried_FLAG = True
                break
            string_f_pt_label = oracle.query(X, self.cl_list[q_cl_ind].f_pt_Mi) #this will be a string label, 
            # in which case we convert it on the next iteration, or it will be "quit" in which case, we will break
            #from the loop        
            
            if string_f_pt_label != "quit":
                #process latest label result
                f_pt_label = self.get_integer_label(string_f_pt_label)
                self.process_query(X, q_cl_ind, f_pt_label)
                #query the next cluster
                self.num_queries += 1
#                 if (self.num_queries%1) == 0:
#                     print(f'number of queries {self.num_queries}')

class Cluster:
    # NOTE: clusters have 2 mutually exclusive pools of points:
    #  l_pts - labeled points, known for sure
    #  o_pts - pts in this cluster, presumed to have this label due to clustering
    #       - if there are no o_pts, try to return -1 when asked for one
    #  f_pt_Mi - index of one of the o_pts, farthest from the l_pts
    #  f_pt_dist - dist to f_pt
    # NOTE: -1 for f_pt_Mi or o_pt_Mi indicates invalid index (possibly because we're down to cluster of 
    # size 1!)
    def __init__(self,X,label,l_pts_Mi,o_pts_Mi):
        self.label = label
        self.l_pts_Mi = l_pts_Mi # a list of M_inds to pts known to have this label
        self.o_pts_Mi = o_pts_Mi # a list of M_inds to pts labelled by clustering
        self.update_far_point(X)# this creates self.f_pt_Mi and self.f_pt_dist
        
    def summary(self):
        print('label: ',self.label,'l pts: ',self.l_pts_Mi,' f_pt index: ',self.f_pt_Mi,
              'number of o_pts: ',len(self.o_pts_Mi))
        
    def update_far_point(self, X):
        # parameters:
        #     l_pts_Mi - indices of pts in data of 'labeled' pts
        #     o_pts_Mi - indices of pts in data of 'other' pts
        # the "far point" - i.e. the point out of all the "other"
        # pts pool that is furthest from the "labeled" pts, in a max min sense
        # (i.e.the minimum distance from the far point to any of the labeled points is >= the minimum distance 
        # from any of the other points to the labeled points)
        # Updates:
        #     f_pt_Mi- index of far point in data
        #     f_pt_dist - max min dist of far pt from labeled pts
        #    NOTE: if there are no o_pts, this cluster is "done"and f_pt_Mi = -1, and f_pt_dist = 0 
        # algorithm:
        #    construct matrix of dists ("tmp_dist_matrix") where rows are l_pts, cols are o_pts
        #    find min of each column - that's the distance from a given o_pt to the nearest l_pt, so that's 
        #      the "worst" case for that o_pt
        #    store all those min's in a row vector
        #    then find the max of that vector - that goes with the o_pt that is the far point

        if not self.o_pts_Mi:  # this means there are no unknown/unlabelled points (ie o pts) in this cluster, so
            # there will be no far pt
            self.f_pt_Mi = -1
            self.f_pt_dist = 0
        else:
            self.l_pts = X[self.l_pts_Mi,:]
            # if only 1 index array shape will be wrong so check and reshape if necessary
            if len(self.l_pts.shape) == 1:
                self.l_pts = np.reshape(self.l_pts,(1,-1))
            self.o_pts = X[self.o_pts_Mi,:]
            if len(self.o_pts.shape) == 1:
                self.o_pts = np.reshape(self.o_pts,(1,-1))
            tmp_dist_matrix = -1*np.ones((len(self.l_pts),len(self.o_pts))) # l pts rows, o pts cols
            for r in range(len(self.l_pts)):
                tmp_dist_matrix[r,:] = self.find_1pt_dists(self.l_pts[r,:],self.o_pts).transpose()
            # find min dist in each column
            min_dists_row = np.amin(tmp_dist_matrix,axis = 0)
            # then find index of max in resulting row
            f_pt_local_ind = np.argmax(min_dists_row)
            self.f_pt_Mi = self.o_pts_Mi[f_pt_local_ind]
            self.f_pt_dist = np.max(min_dists_row)
    
    def incorporate_far_point(self, X):
        # remove f_pt from o_pts
        self.o_pts_Mi.remove(self.f_pt_Mi)
        # add f_pt to l_pts
        self.l_pts_Mi.append(self.f_pt_Mi)
        # calc new f_pt, f_pt_dist
        self.update_far_point(X)
           
    def split(self,X, f_pt_label, engine):
        #params:
            #np.array X: the data
            #int f_pt_label: the new label for the farpoint
            #templates.Cluster_Engine engine: A clusterer that adheres to the interface provided in templates
        #returns:
            #tuple<Cluster, Cluster>: the farpoint cluster, and the non farpoint cluster.
            
        num_l_pts_Mi = len(self.l_pts_Mi)
        
        #  reconstruct an array w/ f_pt_Mi, followed by l_pts_Mi, followed by o_pts_Mi
        # need o_pts WITHOUT f_pt        
        o_pts_Mi_wo_fp = self.o_pts_Mi.copy()
        o_pts_Mi_wo_fp.remove(self.f_pt_Mi)
        # then use o_pts_Mi_wo_fp for o_pts.
        all_pts_Mi = np.array([self.f_pt_Mi] + self.l_pts_Mi + o_pts_Mi_wo_fp)
        
        # get corresponding actual data
        all_pts = X[all_pts_Mi,:]
        
        # so now have all_pts arranged so that
        # all_pts[0] == f_pt
        # all_pts[1:num_l_pts_Mi+1] == l_pts
        # all_pts[num_l_pts_Mi+1:] == o_pts
        engine.fit(all_pts, num_l_pts_Mi)
        clusterer_labels = engine.labels_
        
        o_pt_clusterer_labels = np.array(clusterer_labels[num_l_pts_Mi+1:])
        f_pt_local_label = clusterer_labels[0]
        fp_cl_local_o_pt_inds = np.where(o_pt_clusterer_labels == f_pt_local_label)[0]
        non_fp_cl_local_o_pt_inds = np.where(o_pt_clusterer_labels != f_pt_local_label)[0]
        o_pts_Mi_arr = np.array(o_pts_Mi_wo_fp) #=all_pts[num_l_pts_Mi+1:]
        fp_cl_o_pt_Mi = list(o_pts_Mi_arr[fp_cl_local_o_pt_inds])
        non_fp_cl_o_pt_Mi = list(o_pts_Mi_arr[non_fp_cl_local_o_pt_inds])
        # Now have 4 mut. excl. populations  
        #  orig cl l pts (not f_pt_local_label)
        #  orig cl o pts (not f_pt_local_label)
        #  f_pt (f_pt_local_label) - will become starting l pt of new cluster
        #  new cl o pts (f_pt_local_label)
        # form resulting clusters
        fp_cluster = Cluster(X,f_pt_label,[self.f_pt_Mi],fp_cl_o_pt_Mi)
        non_fp_cluster = Cluster(X,self.label,self.l_pts_Mi,non_fp_cl_o_pt_Mi)
        
#         fp_cluster.summary()
#         non_fp_cluster.summary()
        
        return fp_cluster, non_fp_cluster
    
    def find_1pt_dists(self,pt,other_pts):
        # parameters:
        #     pt - np array shape num_features,
        #     other_pts - np array num_samples x num_features
        # returns:
        #     dists - np array num_samples x 1 - distances between pt and each of the other_pts
        dists = np.zeros((len(other_pts),1))
        for i in range(len(other_pts)):
            dists[i,0] = distance.euclidean(pt, other_pts[i,:])
        return dists


class Oracle(ABC):
   """
   This class is an interface for a class to label points.
   It is designed to be passed into a Partition object.
   The Partition class was designed so that it is possible to stop
   training and pick up with a new Oracle. This allows for alternate
   methods of labeling based on the evolution of the partition.
   """

   @abstractmethod
   def query(self,X, i):
       #params:
           #np.array X: the data
	   #int i: index into the raw data. The point to be queried
       #returns:
           #if to continue
               #str: a label for the given index.
	   #if to end
	       #str: quit
       pass
	   

class Auto_Oracle(Oracle):
    """
    SUMMARY
      Inherits from Oracle
      Implements a version of the query method that returns a label based on a passed in label array
      This class is made to be used only for automated statistical purposes.
      In real applications, the labels will not be known.
    MEMBERS
      np.array y: the labels corresponding to the data that will be passed in
      int n_iters: the number of iteration that we would like to continue for.
      int n_queries_this_iter: the number of calls that have been made to the query method
    """
    
    def __init__(self, y, n_iters, n_classes=None):
        
        # NB: n_iters is the number of labels the oracle gives before "quitting"
        # NB: the oracle returns string labels
        
        #params:
            #np.array y: the labels to the data. The labels must line up with the data. This could be an array 
            # of integers, in which case they get recast to strings directly
            # RO QUESTION: Can y be input as an array of strings?....................
            #int n_iters: the number of iterations that we want to run for while stepping through partition runs
	    #int n_classes: the number of classes to be found before terminating.
	      #if n_classes is not supplied, the oracle will only quit at n_iters - this is accomplished
          # by setting n_classes = None.  It does allow you to continue past the actual number of 
          # classes if you want to for classification accuracy
        #Postconditions:
            #members have been initialized
        
        self.y = y
        self.n_iters = n_iters
        self.n_classes = n_classes
        self.n_queries_this_iter = 0
        self.class_v_query_dict = {}
        self.class_v_i_dict = {} #used to remember the index of the class that was queried
    
    def query(self, X, i):
       #params:
           #np.array X: the data. Note, in this case the data is not important because we have corresponding labels. 
           #int i: index into the raw data. The point to be queried
       #returns:
           #if to continue
               #str: a label for the given index.
           #if to terminate
               #str: quit
        if self.n_queries_this_iter < self.n_iters and (not self.n_classes or len(self.class_v_query_dict) < self.n_classes):
            self.n_queries_this_iter += 1
            label = str(self.y[i])
            #if new class, record when you found it, and index in X
            if label not in self.class_v_query_dict:
                self.class_v_query_dict[label] = self.n_queries_this_iter
                self.class_v_i_dict[label] = i
            return label
        return "quit"
    
    def reset(self):
        # added by Ro to allow stepping through partition runs
        self.n_queries_this_iter = 0
        self.class_v_query_dict = {}

        
class Cluster_Engine(ABC):
    """
    This class is an interface for a class to assist in the spliting process 
    that occurs when a new farpoint is found. It is passed to a Partion during
    the fit process and is then internally passed to the cluster to be split.
    Note that the fit method of this class will be called multiple times on 
    different sets of data. The previous labels_ should be overwritten between
    fits.
    """

    @abstractmethod
    def fit(X, n_labels):
        #params:
	    #np.array X: ordered array of the points in the cluster such that:
	      # X[0] == f_pt
              # X[1:num_l_pts_Mi+1] == l_pts
              # X[num_l_pts_Mi+1:] == o_pts
        #Postconditions:
              #self.labels_ is an ordered list of labels into X
        pass
    
class COPK_Engine(Cluster_Engine):
    """
    SUMMARY
      Inherits from templates.Cluster_Engine
      This class implements a version of COPKMeans that can be found here
      https://github.com/datamole-ai/active-semi-supervised-clustering
      and formats it to be used in the partion.
   MEMBERS
     np.array labels_: labels that index into the data passed in fit.
   """

    def __init__(self):
        self.labels_ = np.nan

    def fit(self, X, n_labels):
        #params:
	    #np.array X: ordered array of the points in the cluster such that:
	      # X[0] == f_pt
              # X[1:num_l_pts_Mi+1] == l_pts
              # X[num_l_pts_Mi+1:] == o_pts
	    #int n_labels: the number of labeled points in the array
        #Postconditions:
              #self.labels_ is an ordered list of labels into X
        n_seeds = n_labels+1
        seed_pts = X[:n_seeds,:]
	#The set of can't link constraints is the cartesian product of {f_pt + l_pt} and itself
        cant_links = list(combinations(range(n_seeds), 2))
        clusterer=COPKMeans(n_clusters=n_seeds)
        clusterer.fit(X,y = seed_pts, ml=[], cl=cant_links)
        self.labels_ = clusterer.labels_
        
class CA_Engine(Cluster_Engine):
    """
    SUMMARY
      Inherits from templates.Cluster_Engine
      This assigns unlabeled points (aka "other") to either the farpoint or l-point cluster 
      based on iterating DBSCAN with epsilon parameter increasing/decreasing by powers of 2 until constraints
      are violated/not violated
   MEMBERS
     np.array labels_: labels that index into the data passed in fit.
   """
    def __init__(self):
        self.labels_ = np.nan
        self.min_samples = 1 # setting to 1 in DBSCAN makes it equivalent to single-link agglomerative clustering
        
    def constraints_violated(self,cl_labels,num_l_pts):
        fp_label = cl_labels[0]
        return fp_label in cl_labels[1:num_l_pts+1] 
    
    def epsilon_check(self,eps,X,num_l_pts):
        cl = DBSCAN(eps, self.min_samples).fit(X)
        too_high_flag = self.constraints_violated(cl.labels_,num_l_pts)
        return cl,too_high_flag
        
    def fit(self, X, num_l_pts):
        #params:
	    #np.array X: ordered array of the points in the cluster such that:
            # X[0] == f_pt
            # X[1:num_l_pts_Mi+1] == l_pts
            # X[num_l_pts_Mi+1:] == o_pts
	    #int num_l_pts: the number of labeled points in the array (not including f_pt)
        #Postconditions:
              #self.labels_ is an ordered list of (local) labels into X
            
        # guess initial epsilon
#         eps = 1 
        # try dist from fp to its nn    
        eps = distance.cdist(X[0].reshape((1,-1)), X[1:]).min(axis=1)[0]

        # do initial clustering to determine whether to go up or down
        cl,too_high_flag = self.epsilon_check(eps,X,num_l_pts)
        # if constraints are violated need to go down, otherwise up
        loop_flag = True
        if not too_high_flag: # so starting out below
            while loop_flag:            
                if not too_high_flag: 
                    cl_L = cl
                    eps_L = eps
                    eps *= 2
                    cl,too_high_flag = self.epsilon_check(eps,X,num_l_pts)
                else:
                    cl_U = cl
                    eps_U = eps
                    loop_flag = False
        else: # starting out above
            while loop_flag:            
                if too_high_flag: 
                    cl_U = cl
                    eps_U = eps
                    eps /= 2
                    cl,too_high_flag = self.epsilon_check(eps,X,num_l_pts)
                else:
                    cl_L = cl
                    eps_L = eps
                    loop_flag = False
        # now have an eps above and an eps below
        # eps_U > T and eps_L < T where T is constraint violation threshold
        
        # now split epsilon difference by halves, keep going until we have a solution where:
        # 1: clustering is below threshold, and 2: clustering has stopped changing
        loop_flag = True
        ARIS_Threshold = 0.95
        MAX_NUM_DBS = 10
        num_DBS = 0
        while loop_flag:
            eps_N = (eps_U - eps_L)/2 + eps_L
#             print(eps_L,eps_U,eps_N)
            cl_N,too_high_flag = self.epsilon_check(eps_N,X,num_l_pts)
            num_DBS += 1
            if too_high_flag:
                eps_U = eps_N
                # eps_L still same
                if num_DBS > MAX_NUM_DBS:
                    loop_flag = False
                    cl = cl_L
#                     print('too many DBSCANs - bail')
            else:
                ARIS = adjusted_rand_score(cl_N.labels_, cl_L.labels_)
#                 print(f'ARIS {ARIS}')
                if ARIS >= ARIS_Threshold:
                    loop_flag = False
                    cl = cl_N
#                     print('ARIS threshold achieved')
                else:
                    eps_L = eps_N
                    # eps_U still same
                    cl_L = cl_N
                if num_DBS > MAX_NUM_DBS:
                    loop_flag = False
                    cl = cl_L
#                     print('too many DBSCANs - bail')
                    
        self.labels_ = cl.labels_
        
        # Nearest neighbor orphan assignment!
        # get local labels of fp and l_pts
        seed_pt_labels = np.unique(self.labels_[:num_l_pts+1])
        # identify indices of orphans
        orphan_inds = []
        for ind in range(num_l_pts+1,X.shape[0]):
            if self.labels_[ind] not in seed_pt_labels:
                orphan_inds.append(ind)
#         print(f'number orphans {len(orphan_inds)}')
        seed_pts = X[:num_l_pts+1]
        orphan_pts = X[orphan_inds]
        closest_seed_inds = distance.cdist(seed_pts, orphan_pts).argmin(axis=0)
        self.labels_[orphan_inds] = self.labels_[closest_seed_inds]
        

class NQ_Engine(Cluster_Engine):
    """
    SUMMARY
      Inherits from templates.Cluster_Engine
      This assigns unlabeled points (aka "other") to the cluster containing the nearest queried (farpoint or l-point) point
   MEMBERS
     np.array labels_: labels that index into the data passed in fit.
   """

    def __init__(self):
        self.labels_ = np.nan
  
    def fit(self, X, n_labels):
        
        #params:
	    #np.array X: ordered array of the points in the cluster such that:
            # X[0] == f_pt
            # X[1:num_l_pts_Mi+1] == l_pts
            # X[num_l_pts_Mi+1:] == o_pts
	    #int n_labels: the number of labeled points in the array
        #Postconditions:
              #self.labels_ is an ordered list of (local) labels into X
        n_seeds = n_labels+1
        seed_pts = X[:n_seeds,:]
        n_pts = len(X)
        # create distance array with pts in rows, cl centers in cols
        cl_dists = -1*np.ones((n_pts,n_seeds))
        for n in range(n_seeds):
            cl_dists[:,n] = np.squeeze(self.find_1pt_dists(seed_pts[n,:],X))
        self.labels_ = np.argmin(cl_dists,axis = 1)

    def find_1pt_dists(self,pt,other_pts):
        # parameters:
        #     pt - np array shape num_features,
        #     other_pts - np array num_samples x num_features
        # returns:
        #     dists - np array num_samples x 1 - distances between pt and each of the other_pts
        dists = np.zeros((len(other_pts),1))
        for i in range(len(other_pts)):
            dists[i,0] = distance.euclidean(pt, other_pts[i,:])
        return dists



def main():

    # generate toy data set 
    # NUM_CLASSES = 3
    num_pts = 1000
    mean = (1, 2)
    cov = [[1, 0], [0, 1]]
    X_c1 = np.random.multivariate_normal(mean, cov, (num_pts))
    y_c1 = 0*np.ones((len(X_c1))).astype(int)
    num_pts = 1000
    mean = (10, 2)
    cov = [[1, 0], [0, 1]]
    X_c2 = np.random.multivariate_normal(mean, cov, (num_pts))
    y_c2 = 1*np.ones((len(X_c2))).astype(int)
    num_pts = 11
    mean = (5,-5)
    cov = [[1, 0], [0, 1]]
    X_c3 = np.random.multivariate_normal(mean, cov, (num_pts))
    y_c3 = 2*np.ones((len(X_c3))).astype(int)
    X = np.concatenate((X_c1,X_c2,X_c3))
    y = np.concatenate((y_c1,y_c2,y_c3)) 
    
    
    # HYPERPARAMETERS
    MAX_NUM_QUERIES = 10
    NUM_QUERIES_PER_ITERATION = 1

    # choose one or the other
#     SPLIT_METHOD = "NQ" # Nearest_Queried
    # SPLIT_METHOD = "COPKM" #'COPK_Means'
    SPLIT_METHOD = "CA" #'Single_Link'

    # choose one or the other
    # SOURCE = 'human'
    SOURCE = 'auto' # get answers directly from pre-existing labels

    # hoping for repeatable behavior
    oracle = Auto_Oracle(y, NUM_QUERIES_PER_ITERATION)
    if SPLIT_METHOD == "COPKM":
        clusterer = COPK_Engine()
    elif SPLIT_METHOD == "NQ":
        clusterer = NQ_Engine() 
    elif SPLIT_METHOD == "CA":
        clusterer = CA_Engine() 
    data_partition = Partition(clusterer)

    for run_num in range(MAX_NUM_QUERIES):    
        oracle.reset()
        data_partition.run(X,oracle)
        predicted_labels = data_partition.predict_labels(X)[0]
        translated_predictions = np.vectorize(data_partition.label_key_.inv.get)(predicted_labels).astype(int)
        toy_data_display(translated_predictions,X)
        print(f'number of classes = {len(data_partition.label_key_)}')

    plt.show()

if __name__ == '__main__':
    
    def toy_data_display(translated_predictions,X):
        c1_inds = np.where(translated_predictions == 0)[0]
        c2_inds = np.where(translated_predictions == 1)[0]
        c3_inds = np.where(translated_predictions == 2)[0]
        fig, axs = plt.subplots(1,1)
        axs.scatter(X[c1_inds,0],X[c1_inds,1],c = 'orange')
        axs.scatter(X[c2_inds,0],X[c2_inds,1],c = 'blue')
        axs.scatter(X[c3_inds,0],X[c3_inds,1],c = 'green')
        axs.set_aspect('equal', 'box')
        axs.set_title('toy data', fontsize=10)
        axs.grid()
        fig.tight_layout()    
    
    main()





