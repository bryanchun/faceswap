import h5py
import numpy as np
import sys

def peak(which, model):
	header = lambda model: 'models/{}'.format(model)
	files = {
		'A': '{}/decoder_A.h5'.format(header(model)),
		'B': '{}/decoder_B.h5'.format(header(model)),
		'E': '{}/encoder.h5'.format(header(model))
	}
	with h5py.File(files[which], 'r') as f:
		for item in f.attrs.keys():
		    print(item + ":", f.attrs[item])
		#for key in list(f.keys()):
		data = np.array(f.get('conv2d_6'))
		print(data)

print(sys.argv)
peak(sys.argv[1], sys.argv[2])
