#!/usr/bin/env python
import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from baseneuron import BaseNeuron

class MorrisLecar(BaseNeuron):

    def __init__(self, n_dict, V_p, dt, debug=False, LPU_id=None,
                 cuda_verbose = False):
        
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        self.num_neurons = len(n_dict['id'])
        self.dt = np.double(dt)
        self.steps = max(int(round(dt / 1e-5)), 1)
        self.debug = debug

        self.ddt = dt / self.steps

        self.V_p = V_p
        self.I = garray.zeros(self.num_neurons, np.double)

        self.n = garray.to_gpu(np.asarray(n_dict['initn'], dtype=np.float64))

        self.V_1 = garray.to_gpu(np.asarray(n_dict['V1'], dtype=np.float64))
        self.V_2 = garray.to_gpu(np.asarray(n_dict['V2'], dtype=np.float64))
        self.V_3 = garray.to_gpu(np.asarray(n_dict['V3'], dtype=np.float64))
        self.V_4 = garray.to_gpu(np.asarray(n_dict['V4'], dtype=np.float64))
        self.V_l = garray.to_gpu(np.asarray(n_dict['V_l'], dtype = np.float64))
        self.V_ca = garray.to_gpu(np.asarray(n_dict['V_ca'], dtype = np.float64))
        self.V_k = garray.to_gpu(np.asarray(n_dict['V_k'], dtype = np.float64))
        self.G_l = garray.to_gpu(np.asarray(n_dict['G_l'], dtype = np.float64))
        self.G_ca = garray.to_gpu(np.asarray(n_dict['G_ca'], dtype = np.float64))
        self.G_k = garray.to_gpu(np.asarray(n_dict['G_k'], dtype = np.float64))
        self.Tphi = garray.to_gpu(np.asarray(n_dict['phi'], dtype=np.float64))
        self.offset = garray.to_gpu(np.asarray(n_dict['offset'],
                                               dtype=np.float64))

        cuda.memcpy_htod(int(self.V_p), np.asarray(n_dict['initV'],
                         dtype=np.double))
        self.update = self.get_euler_kernel()

        super(MorrisLecar, self).__init__(n_dict, V_p, debug)

    @classmethod
    def initneuron(cls, n_dict, neuronstate_p, dt, debug=False, LPU_id=None,
                   cuda_verbose = False):
        return cls(n_dict, neuronstate_p, dt, debug, cuda_verbose)

    @property
    def neuron_class(self):
        return True

    def post_run(self):
        super(MorrisLecar, self).post_run()

    def update_internal_state(self, synapse_state_p, st=None, logger=None):
        super(MorrisLecar, self).update_internal_state_default(
              synapse_state_p, self.I, st, logger)

    def eval(self, st=None):
        self.update.prepared_async_call(
            self.update_grid, self.update_block, st, self.V_p, self.n.gpudata,
            self.num_neurons, self.I.gpudata, self.ddt*1000, self.steps,
            self.V_1.gpudata, self.V_2.gpudata, self.V_3.gpudata,
            self.V_4.gpudata, self.V_l.gpudata, self.V_ca.gpudata,
            self.V_k.gpudata, self.G_l.gpudata, self.G_ca.gpudata,
            self.G_k.gpudata, self.Tphi.gpudata, self.offset.gpudata)

    def get_euler_kernel(self):
        template = """

    #define NVAR 2
    #define NNEU %(nneu)d //NROW * NCOL


    __device__ %(type)s compute_n(%(type)s V, %(type)s n, %(type)s V_3, %(type)s V_4, %(type)s Tphi)
    {
        %(type)s n_inf = 0.5 * (1 + tanh((V - V_3) / V_4));
        %(type)s dn = Tphi * cosh(( V - V_3) / (V_4*2)) * (n_inf - n);
        return dn;
    }

    __device__ %(type)s compute_V(%(type)s V, %(type)s n, %(type)s I, %(type)s V_1, %(type)s V_2, %(type)s V_L, %(type)s V_Ca, %(type)s V_K, %(type)s g_L, %(type)s g_Ca, %(type)s g_K, %(type)s offset)
    {
        %(type)s m_inf = 0.5 * (1+tanh((V - V_1)/V_2));
        %(type)s dV = (I - g_L * (V - V_L) - g_K * n * (V - V_K) - g_Ca * m_inf * (V - V_Ca) + offset);
        return dV;
    }


    __global__ void
    hhn_euler_multiple(%(type)s* g_V, %(type)s* g_n, int num_neurons,
                       %(type)s* I_pre, %(type)s dt, int nsteps,
                       %(type)s* V_1, %(type)s* V_2, %(type)s* V_3,
                       %(type)s* V_4, %(type)s* V_L, %(type)s* V_Ca,
                       %(type)s* V_K, %(type)s* g_L, %(type)s* g_Ca,
                       %(type)s* g_K, %(type)s* Tphi, %(type)s* offset)
    {
        int bid = blockIdx.x;
        int cart_id = bid * NNEU + threadIdx.x;

        %(type)s I, V, n;

        if(cart_id < num_neurons)
        {
            V = g_V[cart_id];
            I = I_pre[cart_id];
            n = g_n[cart_id];

            %(type)s dV, dn;


            for(int i = 0; i < nsteps; ++i)
            {

               dn = compute_n(V, n, V_3[cart_id], V_4[cart_id], Tphi[cart_id]);

               dV = compute_V(V, n, I, V_1[cart_id], V_2[cart_id], V_L[cart_id], V_Ca[cart_id], V_K[cart_id], g_L[cart_id], g_Ca[cart_id], g_K[cart_id], offset[cart_id]);

               V += dV * dt;
               n += dn * dt;
            }


            g_V[cart_id] = V;
            g_n[cart_id] = n;
        }

    }
    """
    # Used 40 registers, 1024+0 bytes smem, 84 bytes cmem[0],
    # 308 bytes cmem[2], 28 bytes cmem[16]
        dtype = np.double
        scalartype = dtype.type if dtype.__class__ is np.dtype else dtype
        self.update_block = (128, 1, 1)
        self.update_grid = ((self.num_neurons - 1) / 128 + 1, 1)
        mod = SourceModule(template % {"type": dtype_to_ctype(dtype),
                           "nneu": self.update_block[0]},
                           options=self.compile_options)
        func = mod.get_function("hhn_euler_multiple")

        func.prepare('PPiP'+np.dtype(dtype).char+'iPPPPPPPPPPPP')
        #             [np.intp, np.intp, np.int32, np.intp, scalartype,
        #              np.int32, np.intp, np.intp, np.intp, np.intp,
        #              np.intp, np.intp, np.intp, np.intp, np.intp,
        #              np.intp, np.intp, np.intp])

        return func

