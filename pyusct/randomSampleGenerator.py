import numpy as np
import util
import os, sys, glob, json, time
sys.path.append("./pyusct/")
from rfdata import RFdata

LOCAL_PATH = "/media/yuhui/dea78678-112b-4f0f-acbf-4e9d1be35e35/nas/"
MOUNT_PATH = "/run/user/1000/gvfs/smb-share:server=azlab-fs01,share=東研究室/個人work/富井/"

class RandomSampleGenerator(object):
    
    # near_edge: sampling non-edge point nearing to edge point or not
    def __init__(self, input_shape=(16, 256, 200), output_shape=(800), near_edge=False):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.near_edge= near_edge
        return 
    
    def getRandomBalancedSampleFromSim(self, num_sampling_per_trial, sim_path, output_path):
        sim_result_dirs = glob.glob( os.path.join(sim_path, "trial*"))
        sim_result_dirs.sort()
        if not os.path.exists(output_path): 
            os.makedirs(os.path.join(output_path, "input"))   # 入力データ
            os.makedirs(os.path.join(output_path, "output"))  # 出力データ      
            os.makedirs(os.path.join(output_path, "sa"))      # 参照用SA

        print("{} trials".format(len(sim_result_dirs))) 
        cnt = 0
        with open(os.path.join(output_path, "list.csv"), 'w') as outf:
            outf.write('id,source,ix,iy,x,y\n')
            for sim_result_dir in sim_result_dirs:
                cnt = self.getRandomBalancedSampleFromTrial(num_sampling_per_trial, sim_result_dir, [16,256,200], [1,1,1], output_path, outf, cnt)
        return
    
    # return data count 
    def getRandomBalancedSampleFromTrial(self, num_sampling, trial_path, input_shape, interval, output_path, outf, cnt):
        print("loading {}".format(trial_path))
        rf = RFdata(trial_path)
        print("{} loaded".format(trial_path.split("/")[-1]));
        r = np.array(rf.param["ringarray"]["radius"])[0,0] * 0.8
        x_min, y_min = rf.pos2ngrid( np.array([- r/np.sqrt(2), - r/np.sqrt(2)]))
        x_max, y_max = rf.pos2ngrid( np.array([+ r/np.sqrt(2), + r/np.sqrt(2)]))
        radius = 0
        sct_img = rf.medium_sct
        offsets = np.arange(-100, 100, 1)

        # get trial ID
        trialID = trial_path.split("/")[-1]
        # set batch size
        batchSize = 1
        # cnt for each sim
        fileCnt = 0

        # 散乱体領域をすべて抽出（割合が小さいため）
        pos_pos = np.array(np.where(sct_img>0.0)).T
        if self.near_edge==False:
            pos_neg = np.array(np.where(sct_img<1.0)).T
        else:
            print("generating near_neg set")
            # direct = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,-1), (-1,1), (1,-1)]
            pos_neg = set()
            dd = 6
            for iy, ix in pos_pos:
                for yy in range(iy-dd, iy+dd):
                    if (yy >= y_max and yy < y_min): continue
                    
                    xx = ix + int(np.sqrt(dd**2 - (yy-iy)**2))
                    if (xx < x_max and xx >= x_min): 
                        if (sct_img[yy][xx]<1.0) and ((yy, xx) not in pos_neg):
                            pos_neg.add((yy,xx))
                    
                    xx = ix - int(np.sqrt(dd**2 - (yy-iy)**2))
                    if (xx < x_max and xx >= x_min): 
                        if (sct_img[yy][xx]<1.0) and ((yy, xx) not in pos_neg):
                            pos_neg.add((yy,xx))
                                       
            pos_neg = np.array(list(pos_neg))
            
        print("generated pos ndarray with shape={}".format(pos_pos.shape))    
        print("generated near_neg ndarray with shape={}".format(pos_neg.shape))
        
                                       
        print("processing")
        for _ in range(0, num_sampling, batchSize): 
            outFile = trialID + "_{}batch".format(str(batchSize)) + "_{}".format(str(fileCnt).zfill(4))
            fileCnt += 1
            dataX = np.zeros((batchSize, input_shape[0]//interval[0], input_shape[1]//interval[1], input_shape[2]//interval[2]), dtype=np.float32)
            datay = np.zeros(batchSize, dtype=np.float32)
            for index in range(batchSize):
                if cnt % 2 == 0:
                    iy, ix= pos_pos[np.random.randint(0, len(pos_pos)), :]
                else:
                    iy, ix= pos_neg[np.random.randint(0, len(pos_neg)), :]

                _, subset = rf.getPointSubset((ix,iy), offsets)
                ## Attention!
                out_image = sct_img[iy-radius:iy+radius+1, ix-radius:ix+radius+1]

                dataX[index] = subset[::interval[0],::interval[1],::interval[2]]
                datay[index] = out_image
                x, y = rf.ngrid2pos(np.array([iy, ix]))
                outf.write('{0},{1},{2},{3},{4},{5}\n'.format(cnt, trial_path, ix,iy, x, y))
                cnt = cnt+1
                if cnt % 100 == 0: print(cnt)
            np.save(os.path.join(os.path.join(output_path, "input/"), outFile), dataX)
            np.save(os.path.join(os.path.join(output_path, "output/"), outFile), datay)
            # print("{} saved.".format(outFile))
        
        print("Finished")
        return cnt
                  
                  
    def debug(self):
        pass
        
    
    
    