import unittest
from FiniteBuffer import FiniteBuffer
from Data_Stream import *
from Oracle import *
from MNIST_Data_Processing import *


N_REL_CLASSES = 8
VERBOSE_FLAGS = [0]

class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.sparsity_levels = [(1 / int(2 ** n)) for n in range(1,11)]
        # Get data and skew and add relevance
        self.X_skewed, _, _, _,self.y_w_rel = MNIST_setup_for_main( N_REL_CLASSES, VERBOSE_FLAGS)
        # Initialize Oracle and ARED ===================================
        self.data_stream = Data_Stream(self.X_skewed, self.y_w_rel)
        self.oracle = Oracle(self.X_skewed, self.y_w_rel)


        self.buffer_size = 1000
        self.ball_tree_ratio = 0.8
        self.num_ball_trees = 2
        self.l_buf = FiniteBuffer(self.buffer_size, self.ball_tree_ratio, self.num_ball_trees)

    def test_forgetting(self):
        for n in range(self.buffer_size*10):
            self.l_buf.insert_pt(self.data_stream.stream_new_data_point(),'0',  True)
        # check abs_index of l_buf
        self.assertEqual(self.l_buf.max_internal_abs_idx, self.buffer_size * 10 - 1, "failed forgetting test")  # add assertion here


    def test_find_closest_pt_in_BT(self):
        for n in range(self.buffer_size*10):
            stream_pt = self.data_stream.stream_new_data_point()
            if n == 0.9*self.buffer_size*10:
                # save point
                comp_pt = stream_pt
            self.l_buf.insert_pt(stream_pt,'0', True)

        # now find comp_pt in buffer
        closest_pt = self.l_buf.find_closest_pts(comp_pt,1)
        print(closest_pt)
        self.assertTrue((comp_pt ==closest_pt[0][3]).all(),"failed finding closest point")  # add assertion here

    def test_find__closest_pts_in_BT(self):
        k = 5
        for n in range(self.buffer_size*10):
            stream_pt = self.data_stream.stream_new_data_point()
            if n == 0.92*self.buffer_size*10:
                # save point
                comp_pt = stream_pt
            self.l_buf.insert_pt(stream_pt,'0', True)

        # now find closest k comp_pts in buffer
        closest_pts = self.l_buf.find_closest_pts(comp_pt,k)
        for n in range(k):
            print(closest_pts[n][0:3])
        self.assertTrue((comp_pt ==closest_pts[0][3]).all(),"failed finding closest point")  # add assertion here
class TestInit(unittest.TestCase):
    def setUp(self):
        self.FiniteBuffer = FiniteBuffer(10)

    def test_btree_ratio_set_up(self):
        self.assertEqual(self.FiniteBuffer.ball_tree_interval, 10 * .8)

    def test_num_ball_trees_set_up(self):
        self.assertEqual(self.FiniteBuffer.num_ball_trees, 2)

    def test_ball_tree_list_set_up(self):
        self.assertTrue(len(self.FiniteBuffer.ball_trees) == 0 )

    def test_buffer_size_set_up(self):
        self.assertEqual(self.FiniteBuffer.data_circular_buffer.size, 10)
        self.assertEqual(self.FiniteBuffer.label_circular_buffer.size, 10)
        #self.assertEqual(self.FiniteBuffer.cluster_id_circular_buffer.size, 10) # removed feature
        self.assertEqual(self.FiniteBuffer.relevance_circular_buffer.size, 10)
        self.assertEqual(self.FiniteBuffer.buffer_size, 10)

if __name__ == '__main__':
    unittest.main()

