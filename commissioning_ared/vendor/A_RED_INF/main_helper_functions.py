import time
from MNIST_Data_Processing import *
from MNIST_2D_Data_Processing import *
from EMNIST_Data_Processing import *
from NICE_Data_Processing import *
from Parking_Lot_Data_Processing import *
from MVtechAD_Data_Processing import *
from DINOv2_MVtechAD_Processing import *
from DINOv2_VisA import *
from VisA_Data_Processing import visa_setup_for_main
# from Parking_Lot_DAGMM_Data_Processing import *
from dagmm_parking_lot import compute_dagmm_features_parking_lot
from DINOv2_Data_Processing import parking_lot_dino_preprocessed
from DINOv2_EMNIST_Data_Processing import *

def get_data(data_source, N_REL_CLASSES, VERBOSE_FLAGS, seed):
    if data_source == "MNIST":
        X, y_w_rel, sparsity_levels, rel_classes = MNIST_setup_for_main(N_REL_CLASSES, VERBOSE_FLAGS, seed)
    elif data_source == "MNIST_2D":
        X, y_w_rel, sparsity_levels, rel_classes = MNIST_2D_setup_for_main(N_REL_CLASSES, VERBOSE_FLAGS, seed)
    elif data_source == "EMNIST":
        X, y_w_rel, sparsity_levels, rel_classes = EMNIST_setup_for_main(N_REL_CLASSES, VERBOSE_FLAGS, seed)
    elif data_source == "EMNIST_DINO":
        X, y_w_rel, sparsity_levels, rel_classes = EMNIST_dino_setup_for_main(N_REL_CLASSES, VERBOSE_FLAGS, seed)
    elif data_source == "NICE":
        X, y_w_rel, sparsity_levels, rel_classes = generate_synthetic_dataset_with_relevance(N_REL_CLASSES, seed)
    elif data_source == "PARKING_LOT_BASE":
        X, y_w_rel, sparsity_levels, rel_classes = parking_lot_setup_for_main(N_REL_CLASSES, VERBOSE_FLAGS, seed)
    elif data_source == "PARKING_LOT_DAGMM":
        X, y_w_rel, sparsity_levels, rel_classes = compute_dagmm_features_parking_lot(
            N_REL_CLASSES=N_REL_CLASSES,
            USE_PCA=False,  # change anytime!
            PCA_COMPS=1024,
            seed=42 + seed,  # keep consistent with other experiments
            verbose=0 in VERBOSE_FLAGS
        )
    elif data_source == "PARKING_LOT_DINO":
        X, y_w_rel, sparsity_levels, rel_classes = parking_lot_dino_preprocessed(
            N_REL_CLASSES,
            VERBOSE_FLAGS,
            seed
        )
    elif data_source == "MVtechAD":
        X, y_w_rel, sparsity_levels, rel_classes = mvtechad_setup_for_main(
            N_REL_CLASSES,
            VERBOSE_FLAGS,
            seed
        )
    elif data_source == "MVtechAD_DINO":
        X, y_w_rel, sparsity_levels, rel_classes = mvtechad_dino_ae_setup_for_main(
            N_REL_CLASSES,
            VERBOSE_FLAGS,
            seed
        )

    elif data_source == "VisA":
        X, y_w_rel, sparsity_levels, rel_classes = visa_setup_for_main(
            N_REL_CLASSES,
            VERBOSE_FLAGS,
            seed
        )

    elif data_source == "VisA_DINO":
        X, y_w_rel, sparsity_levels, rel_classes = visa_dino_ae_setup_for_main(
            N_REL_CLASSES,
            VERBOSE_FLAGS,
            seed
        )
    else:
        raise ValueError("Invalid data source")

    return X, y_w_rel, sparsity_levels, rel_classes


def set_up_stats(ared):
    start_time = time.time()

    times = [start_time]
    num_correct_queries = []
    num_queries = []
    num_clusters = []
    num_labels = []
    conf_matrices = []
    cumulative_relevants = []


    # DEBUG ONLY
    pt_dists = []
    num_pts_searched_list = []

    return start_time, times, num_correct_queries, num_queries, num_clusters, num_labels, conf_matrices, pt_dists, \
        num_pts_searched_list, cumulative_relevants