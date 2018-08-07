import numpy as np
import util
import cv2
import string
import os, json, glob, sys, time

sys.path.append("./pyusct/")
from rfdata import RFdata


class Compressor(object):
    
    def __init__(self, model_path, model_kind="PCA", input_shape=(16, 256, 200), output_shape=(800)):
        
        self.model_path = model_path
        self.model_kind = model_kind
        self.input_shape = input_shape
        self.output_shape = output_shape
        return 
    
    def genDatasetFromSim(self, sim_path, shape, offset, interval, output_path, fromTrial=0, toTrial=50):
        # get trial dirs
        sim_result_dirs = glob.glob(os.path.join(sim_path, "trial*"))
        sim_result_dirs.sort()
        # check trial range
        minIndex = 0
        maxIndex = len(sim_result_dirs) - 1;
        for i in range(len(sim_result_dirs)):
            tmp = int(sim_result_dirs[i].split("/")[-1].split("_")[-1])
            if tmp >= fromTrial:
                minIndex = i;
                break
        for i in range(len(sim_result_dirs))[::-1]:
            tmp = int(sim_result_dirs[i].split("/")[-1].split("_")[-1])
            if tmp <= toTrial:
                maxIndex = i;
                break
        # main           
        for sim_result_dir in sim_result_dirs[minIndex:maxIndex+1]:
            trial_id = sim_result_dir.split("/")[-1]
            self.genDatasetFromTrial(trial_id, sim_result_dir, shape, offset, interval, output_path) 
            
        return 
        
    def genDatasetFromTrial(self, trial_id, trial_path, shape, offset, interval, output_path):
        out_dir = os.path.join(output_path, trial_id)
        if not os.path.exists(out_dir): 
            os.makedirs(os.path.join(out_dir, "input"))   # 入力データ
            os.makedirs(os.path.join(out_dir, "output"))  # 出力データ
        # dataX: compress based on model high to 800 dims
        if self.model_kind=="PCA": 
            self.preprocess_raw_1m1_trial_PCA_4thread(trial_id=trial_id,
                                             input_path=trial_path, 
                                             model_path=self.model_path,
                                             output_path=os.path.join(out_dir, "input/"),
                                             shape=shape,
                                             offset=offset,
                                             interval=interval
                                            )
         
        elif self.model_kind=="AE":
            self.preprocess_raw_1m1_trial_AE(trial_id=trial_id,
                                             input_path=trial_path, 
                                             model_path=self.model_path,
                                             output_path=os.path.join(out_dir, "input/"),
                                             shape=shape,
                                             offset=offset,
                                             interval=interval
                                            )
        else:
            pass
        
        # save label
        rf = RFdata(trial_path)
        xl, xr = offset[0], offset[0] + shape[0]
        yl, yr = offset[1], offset[1] + shape[1]
        datay = rf.medium_sct[xl:xr,yl:yr]
        outFile = "label_size_512m512.npy"
        np.save(os.path.join(os.path.join(out_dir, "output"), outFile), datay)
        return
    
    def preprocess_raw_1m1_trial_AE(self, trial_id, input_path, model_path, output_path, shape, offset, interval):
        #trial_id = sys.argv[1]
        print("{} Process of {} start.".format(time.ctime(), trial_id)) 
        # debug
        print(input_path)
        print(output_path)
        print(model_path)
        
        # create indices
        indices = self.generate_indices(shape, offset)  

        ## Initial RFdata
        rf = RFdata(input_path)
        print("raw data loaded.")

        ## Load model
        import torch
        from torch.autograd import Variable
        from AE import Autoencoder
        
        model = Autoencoder().cuda()
        model.load_state_dict(torch.load(model_path))
        print("AE model loaded.")

        ## Define transfer function
        def dimension_reduce_rf_point(rf, ix, iy):
            offsets = np.arange(-100, 100, interval[2])
            _, subset = rf.getPointSubset((ix,iy), offsets)
            # have to be a parameter
            return subset[::interval[0],::interval[1],:]
        
        ## main
        batch_size = 512*512 // 4
        res = np.empty((batch_size, 800))
        cnt = 0
        print("processing")
        
        for (ix, iy) in indices:
            data = np.transpose(dimension_reduce_rf_point(rf, ix, iy), (2,0,1))[np.newaxis,:,:,:]
            data = Variable(torch.from_numpy(data)).cuda().float()
            tmp, _, _, _ = model.encode(data)
            res[cnt%batch_size] = tmp.detach().cpu().numpy()
            cnt += 1
            if (cnt % 10000 == 0):
                print("{} points completes {}".format(cnt, time.ctime()))
            if ((cnt+1) % batch_size == 0):
                np.save(output_path + "part{}_size{}.npy".format((cnt+1)//batch_size, batch_size), res)
                res = np.empty((batch_size, 800))

        print("{}: completed.".format(time.ctime()))
        return
    
    
    # can switch to 3m3
    def preprocess_raw_1m1_trial_PCA_4thread(self, trial_id, input_path, model_path, output_path, shape, offset, interval):
        #trial_id = sys.argv[1]
        print("{} Process of {} start.".format(time.ctime(), trial_id)) 

        # debug
        print(input_path)
        print(output_path)
        print(model_path)

        # create indices
        indices = self.generate_indices(shape, offset)
        
        # 

        import threading
        class myThread (threading.Thread):
            def __init__(self, threadID, input_path, output_path, model_path, indices, interval, func):
                threading.Thread.__init__(self)
                self.threadID = threadID
                self.input_path = input_path
                self.output_path = output_path
                self.model_path = model_path
                self.indices = indices
                self.interval = interval
                self.func = func
            def run(self):
                self.func( self.threadID,
                      self.input_path,
                      self.output_path,
                      self.model_path,
                      self.indices,
                      self.interval,
                     )
        # multiply threads
        batch_size = shape[0] * shape[1] // 4

        thread1 = myThread(0, input_path, output_path, model_path, indices[:batch_size], interval, self.batch_from_sim_to_pca_1m1)
        thread2 = myThread(1, input_path, output_path, model_path, indices[batch_size:batch_size*2], interval, self.batch_from_sim_to_pca_1m1)
        thread3 = myThread(2, input_path, output_path, model_path, indices[batch_size*2:batch_size*3], interval, self.batch_from_sim_to_pca_1m1)
        thread4 = myThread(3, input_path, output_path, model_path, indices[batch_size*3:batch_size*4], interval, self.batch_from_sim_to_pca_1m1)

        thread1.start()
        time.sleep(63)
        thread2.start()
        time.sleep(63)
        thread3.start()
        time.sleep(63)
        thread4.start()
        thread1.join()
        thread2.join()
        thread3.join()
        thread4.join()

        print("Exiting Main Thread")

        print("{} Process of {} completed.".format(time.ctime(), trial_id))
        print("\n")
        return


    def batch_from_sim_to_pca_1m1(self, fid, input_path, output_path, model_path, indices, interval):

        sfid = str(fid).zfill(3)

        print("{}  Thread {}: start".format(time.ctime(), sfid))

        ## Initial RFdata
        rf = RFdata(input_path)
        print("raw data loaded.")

        ## Load model
        from sklearn.externals import joblib
        from sklearn.decomposition import PCA

        pca = joblib.load(model_path) 
        print("Thread {}: PCA model loaded.".format(sfid))



        ## Define transfer function
        def dimension_reduce_rf_point(rf, ix, iy):
            offsets = np.arange(-100, 100, interval[2])
            _, subset = rf.getPointSubset((ix,iy), offsets)
            # have to be a parameter
            return subset[::interval[0],::interval[1],:]

        res = np.empty((len(indices), 800))
        cnt = 0
        print("Thread {}: processing".format(sfid))
        for (ix, iy) in indices:
            res[cnt] = pca.transform(dimension_reduce_rf_point(rf, ix, iy).reshape(1, -1))
            cnt += 1
            if (cnt % 10000 == 0):
                print("Thread {}: {} points completes {}".format(sfid, cnt, time.ctime()))

        print("Thread {}: Saving file.".format(sfid))
        np.save(output_path + "part{}_size{}.npy".format(sfid, cnt), res)
        print("Thread {}: File saved.".format(sfid))

        print("{}  Thread {}: completed.".format(time.ctime(), sfid))
        return
    
    def generate_indices(self, shape, offset):
        indices = np.indices((shape[0], shape[1]))
        indices[0] += offset[0]
        indices[1] += offset[1]
        indices = indices.transpose(1,2,0)
        indices = indices.reshape(-1, 2)    
        
        return indices
        
        
    def debug(self):
        pass
    