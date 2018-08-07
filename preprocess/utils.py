import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def hist_evaluate(data, threshold=0):
#     print("the number of elements <= {} : {}".format(threshold, data[data<=threshold].shape))
#     print("the number of elements > {} : {}".format(threshold, data[data>threshold].shape))
#     fig = plt.figure()
#     ax1 = fig.add_subplot(211)
#     ax2 = fig.add_subplot(212)
#     sns.distplot(data[data<=threshold].flatten(), ax=ax1)
#     sns.distplot(data[data>threshold].flatten(), ax=ax2)
    sns.distplot(data.flatten(), kde=False)
    
def heatmap_fixT(data, shape=None, T=7):
    data = data[T, :, :].T
    print(data.shape)
    if shape is None:
        shape = np.divide(data.shape, 50)
        shape[0] *= 1.3
    fig, ax = plt.subplots()
    fig.set_size_inches(shape)
    sns.heatmap(data, xticklabels='None', yticklabels='None', cmap="hot", cbar=True, ax=ax)