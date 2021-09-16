
import numpy as np

def find_correct_lobe(img):
  """
  Here the objective of this function is too
  get the bounding box around the most effected lobe of the X-Ray.
  As we know that Pneumoonia can be present anywhere in the lungs
  but the purpose of this function is to get the most effected region
  in the CXR.
  """
  output = []
  im = img.copy()
  r,c,_ = im.shape
  s = []
  im[:,c//2:c,:] = np.zeros((r,c//2,3))
  s.append(im[20:110,:,:].sum())
  im = img.copy()
  im[:,0:c//2,:] = np.zeros((r,c//2,3))
  s.append(im[20:110,:,:].sum())
  im = img.copy()
  box = np.zeros((r,c//2,3))
  p = []
  box2 = np.zeros((r//2,c//2,3))
  if np.array(s).argmax():
    box[2:r-2,2:c//2-2,:] = im[2:r-2,c//2+2:c-2,:]
    im[:,c//2:c,:] = box
    output.append(im)
    im = img.copy()
    im[:,0:c//2,:] = np.zeros((r,c//2,3))
    im[r//2:r,c//2:c,:] = box2
    p.append(im[20:,:,:].sum())
    im = img.copy()
    im[:r,0:c//2,:] = np.zeros((r,c//2,3))
    im[0:r//2,c//2:c,:] = box2
    p.append(np.square(im[r//2:110,:,:]).sum())
    im = img.copy()

    if np.array(p).argmax(): # giving box to the effected lower lobe
      box2[2:r//2-2,2:c//2-2,:] = im[r//2+2:r-2,c//2+2:c-2,:]
      im[r//2:r,c//2:c,:] = box2
      output.append(im)
    else: # giving box to the effected lower lobe
      box2[2:r//2-2,2:c//2-2,:] = im[2:r//2-2,c//2+2:c-2,:]
      im[0:r//2,c//2:c,:] = box2
      output.append(im)

  else:
    box[2:r-2,2:c//2-2,:] = im[2:r-2,2:c//2-2,:]
    im[:,0:c//2,:] = box
    output.append(im)
    #Below is the code for finding the right upper lobe
    im = img.copy()
    im[:r,c//2:c,:] = np.zeros((r,c//2,3))
    im[r//2:r,0:c//2,:] = box2
    p.append(im[20:,:,:].sum())
    #Below is the code for finding the right lower lobe
    im = img.copy()
    im[:r,c//2:c,:] = np.zeros((r,c//2,3))
    im[0:r//2,0:c//2,:] = box2
    p.append(np.square(im[0:110,:,:]).sum())
    im = img.copy()

    if np.array(p).argmax(): # giving box to the effected lower lobe
      box2[2:r//2-2,2:c//2-2,:] = im[r//2+2:r-2,2:c//2-2,:]
      im[r//2:r,0:c//2,:] = box2
      output.append(im)
    else: # giving box to the upper lobe
      box2[2:r//2-2,2:c//2-2,:] = im[2:r//2-2,2:c//2-2,:]
      im[0:r//2,0:c//2,:] = box2
      output.append(im)

  return  output[1]

