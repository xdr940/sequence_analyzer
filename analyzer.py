from utils.yaml_wrapper import YamlHandler
from utils.basic import readlines
from multiprocessing import Pool
from mpl_toolkits.mplot3d import Axes3D
from path import Path
from utils.layers import Photometor
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter



def img2hist( img):
    bins = np.histogram(img, bins=100)
    return bins[0]
class SequenceAnalyzer():
    def __init__(self,args):
        self.data_path = Path(args['data_path'])
        self.analyzer_wkdir = Path(args['analyzer_wkdir'])

        self.logs = self.analyzer_wkdir/args['logs']
        self.dirs = readlines(self.logs)
        self.pool = Pool(processes=12)
        self.step = args['step']
        self.photometor = Photometor(batch_size=args['batch_size'],
                                     pool=self.pool,
                                     step=self.step)
        self.wkdir = self.analyzer_wkdir/self.logs.stem
        self.wkdir.mkdir_p()


        #photometric raw
        self.photometric_raw_dir = self.wkdir / "photometric_error_maps"
        self.photometric_raw_dir.mkdir_p()
        #hist
        self.error_maps_hist_dir = self.wkdir / 'error_maps_hist'
        self.error_maps_hist_dir.mkdir_p()


    def photometric_error_map(self,visual_save=True):
        '''
        1. 根据dirs得到每个seq的photometric-raw(帧)
        2. 根据raw-frame进行统计得到每个frame 的统计量
        3. 通过统计量seq得到此序列的统计量


        :return:
        '''

        for dir in tqdm(self.dirs):

            files = (self.data_path/dir).files()
            files.sort()
            print('{} :{}'.format(dir,len(files)))
            error_maps = self.photometor(paths=files)
            error_maps =  error_maps.cpu().numpy()
            # photometric_err = np.concatenate([error_maps, -np.ones(self.step)])
            # photometric_errs.append(phot   ometric_err)
            # photometric_err = np.array(photometric_err)
            #
            # np.save(photometric_err_dir / str(seq_p.stem) + '.npy', photometric_err)

            file_name = str(dir).replace('/','_')
            if visual_save:
                dir = self.photometric_raw_dir/file_name
                dir.mkdir_p()
                for idx,item in enumerate(error_maps):
                    plt.imsave(dir/'{:04d}.png'.format(idx), item[0])

                    pass


            np.save(file=self.photometric_raw_dir/file_name,arr = error_maps)
        self.pool.close()
        pass

    def photometric_hist(self):
        in_dir = self.photometric_raw_dir
        files = in_dir.files()
        for file in files:

            npy = np.load(file)#
            hists = []

            for item in npy:
                hist, bins, = np.histogram(item, bins=100)
                hists.append(hist)
            hists = np.array(hists)
            np.save(self.error_maps_hist_dir/"{}.npy".format(file.stem),hists)
            # surfacemap(hists)
            # plt.title(file.stem)
            # plt.show()

            print(file.stem)




    def corr_test(self):
        in_dir = self.wkdir
        # mean = np.loadtxt(in_dir/'mean.txt')
        # median = np.loadtxt(in_dir/'median.txt')
        var = np.loadtxt(in_dir/'var-10.txt')*50-0.8
        Lambda = np.loadtxt(in_dir/'lambda-10.txt')


        df = pd.DataFrame({"Lambda":Lambda,"var":var})
        corrs = df['Lambda'].corr(df['var'])
        plt.plot(Lambda,'r')
        plt.plot(var,'g')
        plt.show()
        print(corrs)
        # plt.xlabel('frames')
        # plt.ylabel('lambda')
        #
        #
        # plt.show()
        print('ok')

    def corr_test2(self):
        x = [1, 2, 3, 4, 5]
        y = [6, 7, 8, 9, 6]
        df = pd.DataFrame({'x': x, 'y': y})
        df.x.corr(df.y)





    def autocorr(self):
        '''
        generate auto correlation line for every files in photometric error map,
        and concat lines to a np-array(.npy) ,output selection file(.csv) for dataset split filtering.
        :return:
        *.npy
        *.csv
        '''

        in_dir = self.photometric_raw_dir
        files = in_dir.files()
        selection_table ={}
        total_acs=[]
        files.sort()
        for file in tqdm(files):
            bins = []
            npy = np.load(file)
            for item in npy:
                bin = img2hist(item)
                bins.append(np.expand_dims(bin, axis=0))
            # bins = np.histogram(npy[0,0], bins=100)
            acs = []
            length = len(bins)
            selection=np.ones([len(npy)+2])
            selection.astype(np.int8)
            for idx in range(1,length - 1):
                acc1 = abs(bins[idx] - bins[idx + 1]).sum()
                acc2 = abs(bins[idx-1] - bins[idx]).sum()
                acc = max(acc1,acc2)
                acs.append(acc/(1024*768))

             # some sequecnes is not match the median of sequence length,e.g., in the end of each blocks.
            if len(acs)<len(npy)-1:
                acs.append(acs[-1])



            for idx,v in enumerate(acs):
                if v > np.median(acs)*2:
                    selection[idx]=0
                    selection[idx-1]=0
                    selection[idx+1]=0

            acs = np.array(acs)
            acs = np.expand_dims(acs,axis=0)
            total_acs.append(acs)
            selection_table[file.stem]=selection
        total_acs = np.concatenate(total_acs,axis=0)
        np.save(self.wkdir/'total_acs.npy',total_acs)
        df =pd.DataFrame(selection_table).T
        df.to_csv(self.wkdir/'selection.csv')#这里scences没有

    def csv_description(self):
        '''
        根据sequence的中间输出, 计算一个frame, 并通过frame进行选择
        :return:
        '''

        in_dir = self.error_maps_hist
        files = in_dir.files()

        selector={}
        for file in files:
            sample = file.stem
            npy = np.load(file)
            selector[sample] = 0
            pass



        for file in files:
            bins = []
            print('ok')





    def draw_photometric_analysis(self):
        base_dir = Path('/home/roit/aws/aprojects/sequence_analyzer/sequences/photometric_error_maps')
        files =  base_dir.files()

        for file in files:
            npy = np.load(file)
            #bins = np.histogram(npy[0,0], bins=100)
            pass
        pass




if __name__ == '__main__':

    args = YamlHandler('./settings.yaml').read_yaml()
    analyzer=SequenceAnalyzer(args)
    # analyzer.photometric_error_map()
    analyzer.photometric_hist()
    # analyzer.draw()
    # analyzer.photometric_stat()
    # analyzer.corr_test()




    # analyzer.draw()