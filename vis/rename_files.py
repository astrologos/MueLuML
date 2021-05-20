import os
path = './data/droptol_coarsesize'
for filename in os.listdir(path):
    _, prefix, num = filename[:-4].split('_')
    num = num.zfill(4)
    new_filename = prefix + "_" + num + ".csv"
    os.rename(os.path.join(path, filename), os.path.join(path, new_filename))
