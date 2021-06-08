from fix_pythonpath import *
import imfusion
import numpy as np

imfusion.init()

for item in imfusion.availableAlgorithms():
    if "IO" in item:
        print(item)

sweep = imfusion.UltrasoundSweep()
tr = imfusion.TrackingStream()
for i in range(10):
    #im = imfusion.SharedImage(imfusion.Image.UBYTE, 10, 10, 1, 1)

    arr = (np.array(np.random.random([1, 50, 50, 1])*100).astype(np.uint8))
    im = imfusion.SharedImage(arr)

    im.spacing = np.array([.5, .5, 1])
    im.setDirtyMem()
    sweep.add(im)
    sweep.setTimestamp(i, 0)

    trans_mat = np.eye(4)
    trans_mat[2, -1] = i

    tr.add(trans_mat, i)
    sweep.addTracking(tr)

vol = imfusion.executeAlgorithm("Ultrasound;Sweep Compounding", [sweep])

save_location = "C:\\Users\\maria\\OneDrive\\Desktop\\john_2.imf"
print("Saving result in: {}".format(save_location))

imfusion.executeAlgorithm('IO;ImFusionFile', [sweep], imfusion.Properties({'location': save_location}))



mysweep = imfusion.open("C:\\Users\\maria\\OneDrive\\Desktop\\john_2.imf")[0]
vol1 = imfusion.executeAlgorithm("Ultrasound;Sweep Compounding", [mysweep])

print()