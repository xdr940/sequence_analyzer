
from utils.basic import readlines
from multiprocessing import Pool
from path import Path
from utils.layers import Photometor
import numpy as np

import matplotlib.pyplot as plt


def img2hist( img):
    bins = np.histogram(img, bins=100)
    return bins[0]
class SequenceAnalyzer():
    def __init__(self,args):
        self.data_path = Path(args['data_path'])
        self.sequences_file = Path(args['sequences_file'])
        self.dirs = readlines(self.sequences_file)
        self.pool = Pool(processes=12)
        self.step = args['step']
        self.photometor = Photometor(batch_size=args['batch_size'],
                                     pool=self.pool,
                                     step=self.step)

        self.analyzer_dir = Path('./'+self.sequences_file.stem)
        self.analyzer_dir.mkdir_p()
    def photometric_analysis(self):
        '''
        1. 根据dirs得到每个seq的photometric-raw(帧)
        2. 根据raw-frame进行统计得到每个frame 的统计量
        3. 通过统计量seq得到此序列的统计量


        :return:
        '''
        self.photometric_raw_dir = self.analyzer_dir/"photometric_error_maps"
        self.photometric_raw_dir.mkdir_p()
        for dir in self.dirs:

            files = (self.data_path/dir).files()
            files.sort()

            error_maps = self.photometor(paths=files)
            # photometric_err = np.concatenate([error_maps, -np.ones(self.step)])
            # photometric_errs.append(photometric_err)
            # photometric_err = np.array(photometric_err)
            #
            # np.save(photometric_err_dir / str(seq_p.stem) + '.npy', photometric_err)

            file_name = str(dir).replace('/','_')

            np.save(file=self.photometric_raw_dir/file_name,arr = error_maps.cpu().numpy())
        self.pool.close()
        pass

    def histogram_analysis(self):
        self.error_maps_hist = self.analyzer_dir/'error_maps_hist'
        self.error_maps_hist.mkdir_p()
        in_dir = Path('/home/roit/aws/aprojects/sequence_analyzer/scripts/sequences/photometric_error_maps')
        files = in_dir.files()

        for file in files:
            bins = []
            npy = np.load(file)
            for item in npy:
                bin = img2hist(item)
                bins.append(np.expand_dims(bin,axis=0))
            # bins = np.histogram(npy[0,0], bins=100)
            np.concatenate(bins,axis=0)
            save_name = file.stem+'_hist'
            np.save(self.error_maps_hist/save_name,bins)

    def direct_histogram_analysis(self):
        pass

    def draw(self,mode='error_maps_hist'):
        if mode == 'error_maps_hist':
            in_dir = self.analyzer_dir/'error_maps_hist'
            for file in in_dir.files():
                npy=np.load(file)





    def draw_photometric_analysis(self):
        base_dir = Path('/home/roit/aws/aprojects/sequence_analyzer/scripts/sequences/photometric_error_maps')
        files =  base_dir.files()

        for file in files:
            npy = np.load(file)
            #bins = np.histogram(npy[0,0], bins=100)
            pass
        pass
