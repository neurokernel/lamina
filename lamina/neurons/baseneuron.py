#!/usr/bin/env python

"""
Base neuron class used by LPU.
"""

import warnings
from abc import ABCMeta, abstractmethod
import os.path
import numpy as np

import pycuda.gpuarray as garray
from pycuda.compiler import SourceModule

from neurokernel.LPU.utils.simpleio import *

class BaseNeuron(object):
    __metaclass__ = ABCMeta

    def __init__(self, n_dict, neuron_state_pointer, debug=False, LPU_id=None,
                 cuda_verbose = False):
        '''
        See initmethod for documentation of arguments
        '''
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []
        
        self.__LPU_id = LPU_id
        
        self.__neuron_state_pointer = neuron_state_pointer
        self.__num_neurons = len(n_dict['id'])
        
        _num_dendrite_cond = np.asarray([n_dict['num_dendrites_cond'][i]
                                         for i in range(self.__num_neurons)],
                                        dtype=np.int32).flatten()
        _num_dendrite = np.asarray([n_dict['num_dendrites_I'][i]
                                    for i in range(self.__num_neurons)],
                                   dtype=np.int32).flatten()

        self.__cum_num_dendrite = garray.to_gpu(np.concatenate((
                                np.asarray([0,], dtype=np.int32),
                                np.cumsum(_num_dendrite, dtype=np.int32))))
        self.__cum_num_dendrite_cond = garray.to_gpu(np.concatenate((
                                np.asarray([0,], dtype=np.int32),
                                np.cumsum(_num_dendrite_cond, dtype=np.int32))))
        self.__num_dendrite = garray.to_gpu(_num_dendrite)
        self.__num_dendrite_cond = garray.to_gpu(_num_dendrite_cond)
        self.__pre = garray.to_gpu(np.asarray(n_dict['I_pre'], dtype=np.int32))
        self.__cond_pre = garray.to_gpu(np.asarray(n_dict['cond_pre'],
                                                  dtype=np.int32))
        self.__V_rev = garray.to_gpu(np.asarray(n_dict['reverse'],
                                                dtype=np.double))
        try:
            assert(isinstance(getattr(self, 'I'), garray.GPUArray))
        except AttributeError, AssertionError:
            self.I = garray.zeros(self.__num_neurons, np.double)
        
        if self.I.size != self.__num_neurons:
            self.I = garray.zeros(self.__num_neurons, np.double)
            warnings.warn('GPUArray representing current in ' +
                          '{}'.format(type(self).__name__) + ' neuron of ' +
                          '{}'.format(self.__LPU_id) + ' LPU has been changed ' +
                          ' due to incompatible size.')
        
        self.__update_I_cond = self.get_update_I_cond_func()
        self.__update_I_non_cond = self.get_update_I_non_cond_func()

        self.__debug = debug
        self.__debug_setup()

    def __debug_setup(self):
        if self.__debug:
            if self.__LPU_id is None:
                self.__LPU_id = "default_LPU"

            i = 0
            while True:
                newfile = "{}_{}{}.h5".format(self.__LPU_id,
                                                 self.__class__.__name__, i)
                if not os.path.isfile(newfile):
                    self.__I_file = h5py.File(newfile, "w")
                    self.__I_file.create_dataset('/array',
                                         (0, self.num_neurons),
                                         dtype=np.float64,
                                         maxshape=(None, self.num_neurons))
                    break
                i += 1

    @abstractmethod
    def initneuron(cls, n_dict, neuron_state_p, dt, debug=False, LPU_id=None,
                   cuda_verbose = False):
        '''
        This method is a wrapper of __init__ method but it is defined
        to enforce a specific constructor interface e.g. constructor
        of this base class does not use dt.

        Every neuron class should setup GPU data structures
        needed by it during initialization and provide their initial
        values.

        n_dict is a dictionary representing the parameters needed
        by the neuron class.
        For example, if a derived neuron class called IAF needs a
        parameter called bias, n_dict['bias'] will be vector containing
        the bias values for all the IAF neurons in a particular LPU.

        In addition if neuron uses base neuron initialization,
        n_dict will also contain:
        1. n_dict['cond_pre'] representing the conductance based synapses
           with connection to neurons represented by this class.
        2. n_dict['reverse'] containing the reverse potentials for the
           conductance based synapses.
        3. n_dict['num_dendrites_cond'] representing the number of dendrites for
           neuron in this class of the conductance type. For eg,
           n_dict['num_dendrites_cond'][0] will represent the number of
           conductance based synapses connecting to the neuron having index 0
           in this object
        4. n_dict['I_pre'] representing the indices of the non-conductance
           based synapses with connections to neurons represented by this
           object , eg:- synapses modelled by filters. This is also includes
           any external input to neurons of this class.
        5. n_dict['num_dendrites_I'] representing the number of dendrites for
           neuron in this class of the non-conductance type. E.g,
           n_dict['num_dendrites_I'][0] will represent the number of
           non conductance based synapses, connecting to the neuron
           having index 0.

        Note that you only need the above information if you plan to override the
        default update_internal_state method.

        neuron_state_p: is an integer representing the initial memory location
        on the GPU for storing the neuron states for this object.
        For graded potential neurons, the data type will be double whereas for
        spiking neurons, it will be int.

        dt: simulation time step

        debug: is a boolean and is intended to be used for debugging purposes.

        LPU_id: an identifier of LPU that will appear in logs so it should be unique
        for LPUs of a specific simulation
        '''
        cls(n_dict, neuron_state_p, debug, LPU_id, cuda_verbose)

    @abstractmethod
    def eval(self):
        '''
        This method should update the neuron states. A pointer to
        the start of the memory located will be provided at time of
        initialization.

        self.I.gpudata will be a pointer to the memory location
        where the input current to all the neurons at each step is updated
        if the child class does not override update_I() method
        '''
        pass

    @abstractmethod
    def update_internal_state(self, synapse_state_pointer, st=None, logger=None):
        '''
        This method should compute the internal state of each neuron
        based on the synapse state.

        BaseNeuron provides an implementation of this method.

        synapse_state_p:
        it will be an integer representing the initial memory
        location on the GPU reserved for the synapse states. The data
        type for synapse states will be double.
        In BaseNeuron implementation it may either
        contain conductances or currents.
        The information needed to compute the input state is provided in the
        dictionary n_dict at initialization.
        '''
        pass

    def update_internal_state_default(self, synapse_state_pointer, internal_state,
            st=None, logger=None):

        internal_state.fill(0)
        if self.__pre.size > 0:
            # synapse_state(I) -> internal_state(I)
            self.__update_I_non_cond.prepared_async_call(
                self.__grid_get_input_I, self.__block_get_input_I, st,
                int(synapse_state_pointer), self.__cum_num_dendrite.gpudata,
                self.__num_dendrite.gpudata, self.__pre.gpudata,
                internal_state.gpudata)

        if self.__cond_pre.size > 0:
            # synapse_state(g) -> internal_state(I)
            self.__update_I_cond.prepared_async_call(
                self.__grid_get_input_cond, self.__block_get_input_cond, st,
                int(synapse_state_pointer), self.__cum_num_dendrite_cond.gpudata,
                self.__num_dendrite_cond.gpudata, self.__cond_pre.gpudata,
                internal_state.gpudata, int(self.__neuron_state_pointer),
                self.__V_rev.gpudata)

        if self.__debug:
            dataset_append(self.__I_file['/array'], internal_state.get().reshape((1, -1)))

    @abstractmethod
    def post_run(self):
        '''
        This method will be called at the end of the simulation.
        '''
        if self.__debug:
            self.__I_file.close()

    def get_update_I_cond_func(self):
        template = """
        #define NUM_NEURONS %(num_neurons)d

        __global__ void get_input(double* synapse, int* cum_num_dendrite,
                                  int* num_dendrite, int* pre, double* I_pre,
                                  double* V, double* V_rev)
        {
            // must use block size (32, 32, 1)
            int tidx = threadIdx.x;
            int tidy = threadIdx.y;
            int bid = blockIdx.x;

            int neuron;

            __shared__ int num_den[32];
            __shared__ int den_start[32];
            __shared__ double V_in[32];
            __shared__ double input[32][33];

            if(tidy == 0)
            {
                neuron = bid * 32 + tidx;
                if(neuron < NUM_NEURONS)
                {
                    num_den[tidx] = num_dendrite[neuron];
                    V_in[tidx] = V[neuron];
                }
            } else if(tidy == 1)
            {
                neuron = bid * 32 + tidx;
                if(neuron < NUM_NEURONS)
                {
                    den_start[tidx] = cum_num_dendrite[neuron];
                }
            }

            input[tidy][tidx] = 0.0;

            __syncthreads();

            neuron = bid * 32 + tidy ;
            if(neuron < NUM_NEURONS)
            {
               int n_den = num_den[tidy];
               int start = den_start[tidy];
               double VV = V_in[tidy];

               for(int i = tidx; i < n_den; i += 32)
               {
                   input[tidy][tidx] += synapse[pre[start + i]] * (VV - V_rev[start + i]);
               }
            }

            __syncthreads();

            if(tidy < 8)
            {
                input[tidx][tidy] += input[tidx][tidy + 8];
                input[tidx][tidy] += input[tidx][tidy + 16];
                input[tidx][tidy] += input[tidx][tidy + 24];
            }

            __syncthreads();

            if(tidy < 4)
            {
                input[tidx][tidy] += input[tidx][tidy + 4];
            }

            __syncthreads();

            if(tidy < 2)
            {
                input[tidx][tidy] += input[tidx][tidy + 2];
            }

            __syncthreads();

            if(tidy == 0)
            {
                input[tidx][0] += input[tidx][1];
                neuron = bid*32+tidx;
                if(neuron < NUM_NEURONS)
                {
                    I_pre[neuron] -= input[tidx][0];
                }
            }
        }
        // can be improved
        """
        mod = SourceModule(template % {"num_neurons": self.num_neurons},
                           options = self.compile_options)
        func = mod.get_function("get_input")
        func.prepare(['PPPPPPP'])
        self.__block_get_input_cond = (32, 32, 1)
        self.__grid_get_input_cond = ((self.num_neurons - 1) / 32 + 1, 1)
        return func

    def get_update_I_non_cond_func(self):
        template = """
        #define NUM_NEURONS %(num_neurons)d

        __global__ void get_input(double* synapse, int* cum_num_dendrite,
                                  int* num_dendrite, int* pre, double* I_pre)
        {
            // must use block size (32, 32, 1)
            int tidx = threadIdx.x;
            int tidy = threadIdx.y;
            int bid = blockIdx.x;

            int neuron;

            __shared__ int num_den[32];
            __shared__ int den_start[32];
            __shared__ double input[32][33];

            if(tidy == 0)
            {
                neuron = bid * 32 + tidx;
                if(neuron < NUM_NEURONS)
                {
                    num_den[tidx] = num_dendrite[neuron];
                }
            }else if(tidy == 1)
            {
                neuron = bid * 32 + tidx;
                if(neuron < NUM_NEURONS)
                {
                    den_start[tidx] = cum_num_dendrite[neuron];
                }
            }

            input[tidx][tidy] = 0.0;

            __syncthreads();
            
            neuron = bid * 32 + tidy;

            if(neuron < NUM_NEURONS){

               int n_den = num_den[tidy];
               int start = den_start[tidy];

               for(int i = tidx; i < n_den; i += 32)
               {
                   input[tidy][tidx] += synapse[pre[start] + i];
               }
            }
            __syncthreads();

            if(tidy < 8)
            {
                input[tidx][tidy] += input[tidx][tidy + 8];
                input[tidx][tidy] += input[tidx][tidy + 16];
                input[tidx][tidy] += input[tidx][tidy + 24];
            }

            __syncthreads();

            if(tidy < 4)
            {
                input[tidx][tidy] += input[tidx][tidy + 4];
            }

            __syncthreads();

            if(tidy < 2)
            {
                input[tidx][tidy] += input[tidx][tidy + 2];
            }

            __syncthreads();

            if(tidy == 0)
            {
                input[tidx][0] += input[tidx][1];
                neuron = bid*32+tidx;
                if(neuron < NUM_NEURONS)
                {
                    I_pre[neuron] += input[tidx][0];
                }
            }

        }
        //can be improved
        """
        mod = SourceModule(template % {"num_neurons": self.num_neurons},
                           options = self.compile_options)
        func = mod.get_function("get_input")
        func.prepare('PPPPP')
        self.__block_get_input_I = (32, 32, 1)
        self.__grid_get_input_I = ((self.num_neurons - 1) / 32 + 1, 1)
        return func
