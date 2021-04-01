from utils.yaml_wrapper import YamlHandler
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.animation import FuncAnimation  # 动图的核心函数
from utils.basic import readlines
import tqdm
from path import Path
import seaborn as sns





class Drawer():
    def __init__(self,args):

        self.data_path = Path(args['data_path'])
        self.analyzer_wkdir = Path(args['analyzer_wkdir'])

        self.logs = self.analyzer_wkdir / args['logs']
        self.dirs = readlines(self.logs)


        self.wkdir = self.analyzer_wkdir / self.logs.stem

        # photometric raw
        self.photometric_raw_dir = self.wkdir / "photometric_error_maps"
        # hist
        self.error_maps_hist_dir = self.wkdir / 'error_maps_hist'


        self.fig, self.ax = plt.subplots()

    def heatmap(self):
        in_dir = self.error_maps_hist_dir
        files = in_dir.files()
        files.sort()
        IS_NORM=True


        for idx,file in enumerate(files):
            if idx not in [4,5,6,9]:
                continue
            plt.figure(figsize=[10,10])
            mat = np.load(file)
            frames,bins = mat.shape
            mat = mat[10:,10:]

            # mat=mat.T
            # mat=1/mat
            mat = np.flipud(mat)
            if IS_NORM:
                max = mat.max()
                min = mat.min()
                mat = (mat-min)/(max-min)

            ax = sns.heatmap(mat,cbar=None,cmap='jet')
            # plt.title(file.stem,fontsize=20)
            # plt.xlabel('error value ', fontsize=20, color='k')  # x轴label的文本和字体大小
            # plt.ylabel('Frames->', fontsize=20, color='k')  # y轴label的文本和字体大小

            print(file.stem)
            plt.xticks([])
            plt.yticks([])

            # plt.yticks([0,10,20,30,40,50])
            plt.show()

    def surfacemap(self):
        in_dir = self.error_maps_hist_dir
        files = in_dir.files()
        files.sort()



        for file in files:
            # plt.clf()
            if file.stem not in ['0000e_sildurs-e']:
                continue

            ax = plt.figure().gca(projection='3d')

            mat = np.load(file)/256/388*100


            mat = mat[:, 10:]
            length = mat.shape[1]
            # mat = np.flipud(mat)
            # Make data.
            y_len, xlen = mat.shape
            Y = np.arange(0, y_len, 1)
            X = np.arange(0, xlen, 1)
            X, Y = np.meshgrid(X, Y)
            Z = mat
            # ax.set_zlim([0, 4000])
            ax.set_xlim([0, 100])

            # Plot the surface.
            ax.plot_surface(X, Y, Z, cmap='jet',
                                   linewidth=1, antialiased=True)
            ax.plot(np.linspace(start=0, stop=length, num=length), np.zeros(length), mat[0], linewidth=5, c='r')


            # Customize the z axis.
            # ax.set_zlim(-1.01, 1.01)
            # ax.zaxis.set_major_locator(LinearLocator(10))
            # ax.zaxis.set_major_formatter(FormatStrFormatter('%.2e'))

            # Add a color bar which maps values to colors.
            ax.set_title(file.stem)

            ax.set_xlabel('Error Value')
            ax.set_ylabel('Frames(n)')
            ax.set_zlabel('Pixels Proportion(%)')

            # fig.colorbar(surf, shrink=0.5, aspect=5)
            # ax.plot()

            plt.show(ax)


    def photometric_stat(self):
        MAP_NORM = False
        in_dir = self.error_maps_hist_dir
        files = in_dir.files()
        scenes = []
        log_data = []
        var_list = []
        mean_list = []
        files.sort()
        for file in files:  # files are trajs
            # if file.stem not in ['sequences_06_image_0']:
            #     continue
            mean = []
            median = []
            npy = np.load(file)  #
            b, _, h, w = npy.shape
            npy = npy.reshape([b, h, w])

            # maps normalization
            maps = []

            if MAP_NORM:
                for map in npy:
                    min = map.min()
                    max = map.max()
                    map = (map - min) / (max - min)
                    maps.append(map)
            else:
                maps = npy
            maps = np.array(maps)
            maps = maps.reshape([b, h * w])
            scenes.append(maps)
            means = np.mean(maps, axis=1)
            vars = np.var(maps, axis=1)
            var_list.append(vars)
            mean_list.append(means)
            hist_var, bins_var, = np.histogram(vars, bins=50)
            log_data.append([bins_var[:-1], hist_var])
            # hist_mean,bins_mean  = np.histogram(means,bins=100)

        print('ok')

        # statical_vars['mean'] = plt.plot(-np.mean(map_item_normal,axis=1))

        # plt.plot(means[0],'r')
        # plt.plot(medians[0],'g')
        print('ok')


    def dynamic_img_draw(self):
        path = Path('/home/roit/datasets/mcrandom')
        logs = Path('/home/roit/datasets/analyzer_sequences/logs/mcrandom.txt')
        sub_dirs = readlines(logs)

        alphas = [0.8,0.6,0.4]
        for sub_dir in sub_dirs:
            dir = path/sub_dir
            #DEBUG
            # if dir.stem not in ['0000e/sildurs-e']:
            #     continue

            files = dir.files()
            files.sort()
            images=[]
            for idx in range(len(files)):
                if idx==250:
                    print('ok')
                    pass
                img = plt.imread(files[idx])
                images.append(img)
                if len(images)>len(alphas):
                    images.pop(0)


                plt.clf()
                plt.xticks([])
                plt.yticks([])


                for jdx,item in enumerate(images):
                    plt.imshow(item,alpha=alphas[jdx])

                plt.title("file: {}, frame:{}".format(dir.relpath(path),idx),fontsize='20')
                plt.pause(0.01)
                plt.draw()











        pass

    def dynamic_histogram_draw(self):

        in_dir = self.photometric_raw_dir
        files = in_dir.files()
        alphas=[1,0.8,0.6,0.4,0.2]
        for file in tqdm(files):
            if file.stem not in ['0001_sildurs-e']:
                continue
            plt.figure()
            plt.grid()
            plt.ylim([0,12000])
            bins = []
            npy = np.load(file)
            length = len(npy)
            lines={}
            for idx in range(len(alphas)-1,length):

                for jdx,item in enumerate(alphas):

                    hist,bins = np.histogram(npy[idx-jdx],bins=100)
                    lines[jdx],= plt.plot(bins[:-1],hist,'r',alpha=item)



                ax = plt.gca()
                plt.title("file: {}, frame: {}".format(file.stem,idx))
                # if idx >250:
                #     plt.show()

                plt.pause(0.1)
                for jdx in lines.keys():
                    ax.lines.remove(lines[jdx])

            # bins = np.histogram(npy[0,0], bins=100)
            # np.concatenate(bins, axis=0)
            # save_name = file.stem
            # np.save(self.error_maps_hist / save_name, bins)

    def single_lines(self):
        in_dir0 = self.analyzer_wkdir/'mcrandom/error_maps_hist'
        in_dir1 = self.analyzer_wkdir/'mcv2/error_maps_hist'
        in_dir2 = self.analyzer_wkdir/'mcv3/error_maps_hist'
        in_dir3 = self.analyzer_wkdir/'mcv5_shaders_comp/error_maps_hist'


        files0= in_dir0.files()
        files0.sort()
        files1 = in_dir1.files()
        files1.sort()
        files2 = in_dir2.files()
        files2.sort()
        files3 = in_dir3.files()
        files3.sort()

        # seq0=np.load(files0[0])/388/256*100
        # seq1=np.load(files0[0])/388/256*100
        # seq2=np.load(files0[1])/388/256*100
        # seq3=np.load(files0[2])/388/256*100

        seq1 = np.load(files3[9]) / 388 / 256 * 100
        seq2 = np.load(files3[5]) / 388 / 256 * 100
        seq3 = np.load(files3[6]) / 388 / 256 * 100
        seq4 = np.load(files3[4]) / 388 / 256 * 100


        frame = 46

        plt.plot(seq1[frame],'r')
        plt.plot(seq2[frame],'g')
        plt.plot(seq3[frame],'b')
        plt.plot(seq4[frame],'c')

        # plt.legend(['$\lambda(49)$','$\lambda(167)$','$\lambda(249)$'])
        plt.legend(['non-blur','low-blur','middle-blur','high-blur'])#motion blur show


        plt.xlabel('error value')
        plt.ylabel('Proportion of pixels (%)')
        plt.grid()
        # plt.title('The histogram of error map in frame sequences')
        plt.title('The histogram of error map in different blur modes')



        # plt.ylim([0, 7])
        plt.show()
        print('ok')




if __name__ == '__main__':
    args = YamlHandler('./settings.yaml').read_yaml()

    drawer = Drawer(args)
    drawer.heatmap()
    # drawer.surfacemap()
    # drawer.photometric_stat()
    # drawer.dynamic_img_draw()
    # drawer.single_lines()