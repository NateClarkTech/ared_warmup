import numpy as np
def skew_weighted_accuracy(y,y_pred):
    # assume we have integer class labels
    # (not assuming class labels are contiguous starting from 0)
    unique_class_labels = list(np.unique(y))
    
    # calculate a frequency list for all classes
    num_samples = y.shape[0]
    skwa = 0
    for l in unique_class_labels:
        # l is label - not label pt
        inds_ls = np.nonzero(y == l)[0]
        num_ls = inds_ls.shape[0]
        num_matches_l = (y_pred[inds_ls] == l).sum()
#         print(num_ls,num_matches_l,round(num_matches_l/num_ls,2))
        skwa += num_matches_l/(num_ls)
    skwa = skwa/(len(unique_class_labels))
    return skwa
#     # for each class, calculate accuracy and weight that by inverse of frequency
#     for l in unique_class_labels:
      
