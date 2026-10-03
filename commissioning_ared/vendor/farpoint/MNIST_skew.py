# MNIST_skew.py
# This is hardcoded to just work w/ MNIST. 

NUM_CLASSES = 10

import numpy as np


def get_MNIST(whos_computer):
    if whos_computer == "Ro":
        npzfile = np.load("/home/johnnybravo/data/mnist.npz")
        X = npzfile['mnist_data']
        y = npzfile['mnist_labels']
    elif whos_computer == 'Noah': #Noah's computer    
        npzfile = np.load("mnist.npz")
        X = npzfile['mnist_data']
        y = npzfile['mnist_labels']
    else:
        response = datasets.fetch_openml("mnist_784")
        X = response['data'].astype(float)
        y = response['target'].astype(int)
        
    return (X,y)
      
# generate skewed data
# NOTE: each class has half prob of next

# Generate skewed data -------------------------------------------------------------------
def skew_data(data, labels):
    # set class frequencies - each class has half prob of next
    skew_vec = np.empty((NUM_CLASSES,))
    for n in range(NUM_CLASSES):
        skew_vec[n] = 2**n
    skew_vec = skew_vec/skew_vec[-1]
#     print(skew_vec)


    shfl_cls = np.arange(NUM_CLASSES) # randomly shuffle class numbers 
    np.random.shuffle(shfl_cls)
#     print(shfl_cls)
    # use all available samples of most represented class (i.e. at skew_vec[9])
    most_fr_class_ind = shfl_cls[-1]
    
    num_samples_in_class = np.empty((NUM_CLASSES),dtype = 'int') # vector that holds number of samples from each class
    n_samples_most_frq_class = len(np.where(labels == most_fr_class_ind)[0])
    num_samples_in_class[-1] = n_samples_most_frq_class
    
    num_samples_this_class = n_samples_most_frq_class
    for n in range(NUM_CLASSES-2,-1,-1):
        num_samples_this_class = int(num_samples_this_class/2)
        num_samples_in_class[n] = num_samples_this_class
    total_num_samples = num_samples_in_class.sum()
                
    skewed_label_list = []
    skewed_data = np.empty((total_num_samples,data.shape[1]))
    class_probs = np.empty(NUM_CLASSES)
        
    loc_in_skewed_data_arr = 0
    for c in range(NUM_CLASSES):
        shfld_c = shfl_cls[c]
        this_c_Mi= np.where(labels == shfld_c)[0]
        n_samples_this_class = len(this_c_Mi)
        class_probs[shfld_c] = skew_vec[c]
        n_samples_skewed = num_samples_in_class[c]
    #     print(n_samples_skewed)
        tmp = [shfld_c]*n_samples_skewed
        skewed_label_list = skewed_label_list + tmp
        sk_inds_c_Mi = np.random.choice(this_c_Mi,n_samples_skewed,replace=False) # these are k random indices out of the total number 
        # of samples with this class label, where k is n_samples_skewed
        skewed_data[loc_in_skewed_data_arr:(loc_in_skewed_data_arr+n_samples_skewed)] = data[sk_inds_c_Mi,:]
        loc_in_skewed_data_arr += n_samples_skewed
    skewed_labels = np.array(skewed_label_list)
    
    return skewed_data,skewed_labels,skew_vec,class_probs
# NOTE: "probs" here in both skew_vec and class_probs are actually 2*real prob. and real prob is really
# off by a small amount from adding up to 1 because we cut off series after 10 terms