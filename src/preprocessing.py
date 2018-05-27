from sklearn.cluster import KMeans
from tqdm import tqdm
import pprint, pickle
import numpy as np
import tensorflow as tf
import numpy as np
import os.path as osp
from sklearn.neighbors import NearestNeighbors

from ThreeDLAPGAN.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from ThreeDLAPGAN.src.point_net_ae import PointNetAutoEncoder
from  ThreeDLAPGAN.src.in_out import  PointCloudDataSet

from  ThreeDLAPGAN.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet

from ThreeDLAPGAN.src.tf_utils import reset_tf_graph

def smart_downsampling(data,pc_count,write_dir = 'log.txt'):
    buf_size=1
    fout = open(write_dir, 'wb', buf_size)
    pclouds_smart = np.empty([data.shape[0],pc_count, data.shape[2]], dtype=np.float32)
    for i,pc in tqdm(enumerate(data)):
        while True:
            try:
                fout.write(str(i)+'\n')
                kmeans  = KMeans(n_clusters = pc_count,n_jobs=-1)
                kmeans.fit(pc)
                pclouds_smart[i] = kmeans.cluster_centers_ 
                break
            except:
                continue
    fout.close()
    return pclouds_smart

def smart_upsampling(data,pc_count,num_neighbors=5,write_dir = 'log.txt'):
    buf_size=1
    fout = open('log.txt', 'wb', buf_size)
    pclouds_smart = np.empty([data.shape[0],data.shape[1]+pc_count, data.shape[2]], dtype=np.float32)
    for i,pc in tqdm(enumerate(data)):
        while True:
            try:
                fout.write(str(i)+'\n')
                knn  = NearestNeighbors(n_neighbors=num_neighbors)

                knn.fit(pc)

                neighbors = knn.kneighbors(pc, return_distance=False)
                pclouds_smart[i,:pc_count] = pc
                pclouds_smart[i,pc_count:] = pc[knn.kneighbors(pc, return_distance=False)].mean(axis=1)
                break
            except:
                continue
    fout.close()
    return pclouds_smart

def read_pickle_data(input='data.pkl'):
    pkl_file = open(input, 'rb')

    pclouds = pickle.load(pkl_file)

    pkl_file.close()
    return pclouds

def pickle_data(pcloud_smart=None,output='data.pkl'):
    output = open(output, 'wb')
    pickle.dump(pcloud_smart, output)
    output.close()
####################################################################################

def Train_AE(train_ae = False,data = None,conf=None,URL = '/home/service/ThreeDLAPGAN/data/single_class_ae/',held_out_data = None):
    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)
    buf_size = 1 # flush each line
    all_pc_data  = PointCloudDataSet(data)
    fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
    if held_out_data is not None:
        held_out_data =  PointCloudDataSet(held_out_data)
    train_stats = ae.train(all_pc_data, conf, log_file=fout, held_out_data = held_out_data )
    fout.close()
    return ae
def Restore_AE(name ='AE512',epoch = 500,conf=None):
    top_dir ='../data/'
    direct =top_dir+name +'/'
    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)
    ae.restore_model(model_path=direct,epoch=epoch)
    return ae

def get_latent(ae,feed_pc):
    latent_codes = ae.transform(feed_pc)
    return latent_codes

def get_encode_decode(ae,feed_pc):
    return ae.decode(ae.transform(feed_pc))