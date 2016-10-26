import caffe
import numpy as np

# put this pyloss.py file under PYTHONPATH, which is under caffe/python folder. 

#import pycuda.autoinit
import pycuda.gpuarray as gpuarray
#from pycuda.compiler import SourceModule
#from pycuda.reduction import ReductionKernel
from pycuda.elementwise import ElementwiseKernel

#import caffe
from caffe import Layer
import caffe.pycuda_util as pu

dtype = np.float32

import skcuda.linalg as linalg
import skcuda.misc as misc


class PcaCudaEuclideanLossLayer(caffe.Layer):
    """
    compute loss between ldr and estimated ldr.  
    input: bottom[0] estimated PCAcoefs--> CRF,  bottom[1] basisPCA, bottom[2] meanPCA, bottom[3] png. 
    """
    
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 4:
            raise Exception("Need crf, basisPCA, meanPCA and ldr to compute distance.")
        self.nPCAcoms = bottom[0].data.shape[1];
        
        #self.bottom23 = bottom[2].data - bottom[3].data;
	
    def reshape(self, bottom, top):

        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[3].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)


    def forward(self, bottom, top):
#        print 'hanli crf forward -- '
#        print 'self.diff.shape: ' + str(self.diff.shape);  # self.diff.shape: (batchsize, 65536)
#        print 'crf bottom[0].data.shape: ' + str(bottom[0].data.shape); #crf bottom[0].data.shape: (batchsize, 11)
#        print 'raw degree bottom[1].data.shape: ' + str(bottom[1].data.shape);  #(batchsize, 65536, 11)
#        print 'png bottom[2].data.shape: ' + str(bottom[2].data.shape);  # (batchsize, 65536)
#        print 'np.dot(bottom[1].data[i,:,:], bottom[0].data[i,:]).shape: ' + str(np.dot(bottom[1].data[0,:,:], bottom[0].data[0,:]).shape); #(65536,)  
#        print 'bottom[2].data[i,:].shape: ' + str(bottom[2].data[0,:].shape);  # (65536,)
        with pu.caffe_cuda_context():
            linalg.init()
            for i in range(self.diff.shape[0]):
                    #a =  bottom[1].data_as_pycuda_gpuarray() 
                    #b =  bottom[0].data_as_pycuda_gpuarray() 
                    a =  bottom[1].data[i,:,:].astype(np.float32);
                    b =  bottom[0].data[i,:].astype(np.float32);
                    ##a = np.asarray(np.random.rand(4, 4), dtype=np.float32)
                    ##b = np.asarray(np.random.rand(4), dtype=np.float32)
                    
                    #a_gpu = gpuarray.GPUArray(a, dtype=np.float32)
                    #b_gpu = gpuarray.GPUArray(b, dtype=np.float32)
                    a_gpu = gpuarray.to_gpu(a)
                    b_gpu = gpuarray.to_gpu(b)
                    c_gpu = linalg.dot(a_gpu, b_gpu)
                    #self.diff[i,:] = c_gpu + bottom[2].data[i,:] - bottom[3].data[i,:];
                    self.diff[i,:] = np.dot(bottom[1].data[i,:,:], bottom[0].data[i,:]) + bottom[2].data[i,:] - bottom[3].data[i,:];
            top[0].data[...] = np.sum(self.diff**2) / bottom[3].num / 2.
            #self.transDiff = np.transpose(self.diff / bottom[3].num); # (65536, 50) 
            a_gpu = gpuarray.to_gpu(self.diff / bottom[3].num)
            at_gpu = linalg.transpose(a_gpu)
            self.transDiff = at_gpu; # (65536, 50)    

    def backward(self, top, propagate_down, bottom):
        
#        self.nPCAcoms = bottom[0].data.shape[1];
        with pu.caffe_cuda_context():
            #for i in range(self.nPCAcoms): 
                #bottom[0].diff[:, i] = np.trace(np.dot( bottom[1].data[:,:,i], self.transDiff ));
            linalg.init()
            for i in range(self.nPCAcoms): 
                ##a =  bottom[1].data[:,:,i].data_as_pycuda_gpuarray() 
                a =  bottom[1].data[:,:,i]
                b_gpu = self.transDiff;
                a_gpu = gpuarray.to_gpu(a)
                c_gpu = linalg.dot(a_gpu, b_gpu)
                d_gpu = linalg.trace(c_gpu)
                bottom[0].diff[:, i] = d_gpu;



class PCAEuclideanLossLayer(caffe.Layer):
    """
    compute loss between ldr and estimated ldr.  
    input: bottom[0] estimated PCAcoefs--> CRF,  bottom[1] basisPCA, bottom[2] meanPCA, bottom[3] png. 
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 4:
            raise Exception("Need crf, basisPCA, meanPCA and ldr to compute distance.")
        self.nPCAcoms = bottom[0].data.shape[1];
        #self.bottom23 = bottom[2].data - bottom[3].data;
	
    def reshape(self, bottom, top):

        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[3].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)


    def forward(self, bottom, top):
#        print 'hanli crf forward -- '
#        print 'self.diff.shape: ' + str(self.diff.shape);  # self.diff.shape: (batchsize, 65536)
#        print 'crf bottom[0].data.shape: ' + str(bottom[0].data.shape); #crf bottom[0].data.shape: (batchsize, 11)
#        print 'raw degree bottom[1].data.shape: ' + str(bottom[1].data.shape);  #(batchsize, 65536, 11)
#        print 'png bottom[2].data.shape: ' + str(bottom[2].data.shape);  # (batchsize, 65536)
#        print 'np.dot(bottom[1].data[i,:,:], bottom[0].data[i,:]).shape: ' + str(np.dot(bottom[1].data[0,:,:], bottom[0].data[0,:]).shape); #(65536,)  
#        print 'bottom[2].data[i,:].shape: ' + str(bottom[2].data[0,:].shape);  # (65536,)
        for i in range(self.diff.shape[0]):
            self.diff[i,:] = np.dot(bottom[1].data[i,:,:], bottom[0].data[i,:]) + bottom[2].data[i,:] - bottom[3].data[i,:];
            #self.diff[...] = np.dot(bottom[1].data, bottom[0].data) - bottom[2].data
        top[0].data[...] = np.sum(self.diff**2) / bottom[3].num / 2.
        self.transDiff = np.transpose(self.diff / bottom[3].num); # (65536, 50) 
#        n = bottom[0].data.shape[1];
#	transRawDegree = np.zeros((n, bottom[1].data.shape[1], bottom[1].data.shape[0]), dtype=np.float32);
#	print transRawDegree.shape
#	print transRawDegree[0, :, :].shape
#	print bottom[1].data[:,:,0].shape
#	for i in range(n):  
#		transRawDegree[i, :, :] = np.transpose( bottom[1].data[:,:,i] );

    def backward(self, top, propagate_down, bottom):
#        self.nPCAcoms = bottom[0].data.shape[1];
        for i in range(self.nPCAcoms): 
            bottom[0].diff[:, i] = np.trace(np.dot( bottom[1].data[:,:,i], self.transDiff ));
    
    
class CRFEuclideanLossLayer(caffe.Layer):
    """
    compute loss between ldr and estimated ldr.  
    input: bottom[0] CRF,  bottom[1] raw degree, bottom[2] png. 
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need crf, hdr and ldr to compute distance.")
        self.ndegree = bottom[0].data.shape[1];
        self.transRawDegree = np.zeros((self.ndegree, bottom[1].data.shape[1], bottom[1].data.shape[0]), dtype=np.float32);
        for i in range(self.ndegree):  
            self.transRawDegree[i, :, :] = np.transpose( bottom[1].data[:,:,i] );
	
    def reshape(self, bottom, top):
        # check input dimensions match
#        if bottom[0].count != bottom[1].count:
#            raise Exception("hdr and ldr must have the same dimension.")
        # check input dimensions match
#        if bottom[2].count != bottom[1].count:
#            raise Exception("weight and Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[2].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)


    def forward(self, bottom, top):
#        print 'hanli crf forward -- '
#        print 'self.diff.shape: ' + str(self.diff.shape);  # self.diff.shape: (batchsize, 65536)
#        print 'crf bottom[0].data.shape: ' + str(bottom[0].data.shape); #crf bottom[0].data.shape: (batchsize, 11)
#        print 'raw degree bottom[1].data.shape: ' + str(bottom[1].data.shape);  #(batchsize, 65536, 11)
#        print 'png bottom[2].data.shape: ' + str(bottom[2].data.shape);  # (batchsize, 65536)
#        print 'np.dot(bottom[1].data[i,:,:], bottom[0].data[i,:]).shape: ' + str(np.dot(bottom[1].data[0,:,:], bottom[0].data[0,:]).shape); #(65536,)  
#        print 'bottom[2].data[i,:].shape: ' + str(bottom[2].data[0,:].shape);  # (65536,)
        for i in range(self.diff.shape[0]):
            self.diff[i,:] = np.dot(bottom[1].data[i,:,:], bottom[0].data[i,:]) - bottom[2].data[i,:];
#        self.diff[...] = np.dot(bottom[1].data, bottom[0].data) - bottom[2].data
        top[0].data[...] = np.sum(self.diff**2) / bottom[2].num / 2.
        self.transDiff = np.transpose(self.diff / bottom[2].num); # (65536, 50) 
#        n = bottom[0].data.shape[1];
#	transRawDegree = np.zeros((n, bottom[1].data.shape[1], bottom[1].data.shape[0]), dtype=np.float32);
#	print transRawDegree.shape
#	print transRawDegree[0, :, :].shape
#	print bottom[1].data[:,:,0].shape
#	for i in range(n):  
#		transRawDegree[i, :, :] = np.transpose( bottom[1].data[:,:,i] );

    def backward(self, top, propagate_down, bottom):
        for i in range(self.ndegree): 
            bottom[0].diff[:, i] = np.trace(np.dot( bottom[1].data[:,:,i], self.transDiff ));
#            copy50times50 = np.dot( bottom[1].data[:,:,i], self.transDiff )
#            bottom[0].diff[:, i] = np.trace(copy50times50);
            
#for i in range(self.ndegree): 
            #copy50times50 = np.dot(self.diff / bottom[2].num,  self.transRawDegree[i, :, :]);            
            
#        n = bottom[0].data.shape[1];
        #transRawDegree = np.zeros((n, bottom[1].data.shape[1], bottom[1].data.shape[0]), dtype=np.float32);
        #for i in range(n):  
            #transRawDegree[i, :, :] = np.transpose( bottom[1].data[:,:,i] );
        #for i in range(self.ndegree): 
            #copy50times50 = np.dot(self.diff / bottom[2].num,  self.transRawDegree[i, :, :]);
#            bottom[0].diff[:, i] = np.trace(copy50times50);
#        print bottom[0].data.shape;  # (50, 11)
#        print 'ndegree is ' + str(n);  # ndegree is 11
#        print 'bottom[0].diff.shape: ' + str(bottom[0].diff.shape);  # bottom[0].diff.shape: (50, 11)
#        print 'bottom[1].data[:,:,i].shape--' + str(bottom[1].data[:,:,0].shape);  # (50, 65536)
#        print 'self.diff / bottom[2].num  .shape--' + str((self.diff / bottom[2].num).shape);  # (50, 65536)
#        print 'np.multiply(bottom[1].data[:,:,i], self.diff / bottom[2].num).shape--' + str((np.multiply(bottom[1].data[:,:,0], self.diff / bottom[2].num).shape));    # (50, 65536)
#        print 'np.sum(np.multiply(bottom[1].data[:,:,i], self.diff / bottom[2].num), axis=1).shape--' + str((np.sum(np.multiply(bottom[1].data[:,:,0], self.diff / bottom[2].num), axis=1).shape));  # (50,)
#        for i in range(n):  
            # element wise multiplication of  diff & raw degree, and then sum together for each degree. 
#            bottom[0].diff[:, i] = np.sum(np.multiply(bottom[1].data[:,:,i], self.diff / bottom[2].num), axis=1); too slow one iteration takes around 10-25 mins? 
#            bottom[0].diff[:, i] = np.trace( np.dot( bottom[1].data[:,:,i], np.transpose(self.diff / bottom[2].num) ) );
            
            
class WeightedEuclideanLossLayer(caffe.Layer):
    """
    Compute the weighted Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need two inputs and one weight to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Two Inputs must have the same dimension.")
        # check input dimensions match
        if bottom[2].count != bottom[1].count:
            raise Exception("weight and Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[2].data * ( bottom[0].data - bottom[1].data )
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            # same as below non-weight loss function, because self.diff contains the weight inside.  based on gradient function, this is correct. 
            bottom[i].diff[...] = sign * self.diff / bottom[i].num


class EuclideanLossLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num
