#!/usr/bin/env python

import os, resource, sys
import argparse

import numpy as np
import networkx as nx

import neurokernel.core_gpu as core
from neurokernel.tools.logging import setup_logger
from neurokernel.tools.timing import Timer
from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor

from neurokernel.LPU.LPU import LPU
import lamina.lamina as lam

import lamina.geometry.hexagon as hx
from lamina.configreader import ConfigReader
from retina.screen.map.mapimpl import AlbersProjectionMap
from retina.NDComponents.MembraneModels.BufferVoltage import BufferVoltage


dtype = np.double
RECURSION_LIMIT = 80000


def setup_logging(config):
    log = config['General']['log']
    file_name = None
    screen = False

    if log in ['file', 'both']:
        file_name = 'neurokernel.log'
    if log in ['screen', 'both']:
        screen = True
    logger = setup_logger(file_name=file_name, screen=screen)


def get_lamina_id(i):
    return 'lamina{}'.format(i)


def add_lamina_LPU(config, i, lamina, manager):
    output_filename = config['Lamina']['output_file']
    gexf_filename = config['Lamina']['gexf_file']
    suffix = config['General']['file_suffix']

    dt = config['General']['dt']
    debug = config['Lamina']['debug']
    time_sync = config['Lamina']['time_sync']

    output_file = '{}{}{}.h5'.format(output_filename, i, suffix)
    gexf_file = '{}{}{}.gexf.gz'.format(gexf_filename, i, suffix)
    G = lamina.get_graph()
    nx.write_gexf(G, gexf_file)

    comp_dict, conns = LPU.graph_to_dicts(G)
    lamina_id = get_lamina_id(i)
    
    extra_comps = [BufferVoltage]
    
    output_processor = FileOutputProcessor(
                            [('V',None)], output_file,
                            sample_interval=1)

    manager.add(LPU, lamina_id, dt, comp_dict, conns,
                output_processors = [output_processor],
                device=i, debug=debug, time_sync=time_sync,
                extra_comps = extra_comps)


def start_simulation(config, manager):
    steps = config['General']['steps']
    with Timer('lamina simulation'):
        manager.spawn()
        manager.start(steps=steps)
        manager.wait()


def change_config(config, index):
    '''
        Useful if one wants to run the same simulation
        with a few parameters changing based on index value

        Need to modify else part

        Parameters
        ----------
        config: configuration object
        index: simulation index
    '''
    if index < 0:
        pass
    else:
        suffixes = ['__{}'.format(i) for i in range(3)]
        values = [5e-4, 1e-3, 2e-3]

        index %= len(values)
        config['General']['file_suffix'] = suffixes[index]
        config['General']['dt'] = values[index]


def get_config_obj(args):
    conf_name = args.config

    # append file extension if not exist
    conf_filename = conf_name if '.' in conf_name else ''.join(
        [conf_name, '.cfg'])
    conf_specname = os.path.join('..', 'template_spec.cfg')

    return ConfigReader(conf_filename, conf_specname)


def main():
    import neurokernel.mpi_relaunch
    # default limit is low for pickling
    # the data structures passed through mpi
    sys.setrecursionlimit(RECURSION_LIMIT)
    resource.setrlimit(resource.RLIMIT_STACK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='default',
                        help='configuration file')
    parser.add_argument('-v', '--value', type=int, default=-1,
                        help='Value that can overwrite configuration '
                             'by changing this script accordingly. '
                             'It is useful when need to run this script '
                             'repeatedly for different configuration')

    args = parser.parse_args()

    with Timer('getting configuration'):
        conf_obj = get_config_obj(args)
        config = conf_obj.conf
        change_config(config, args.value)

    setup_logging(config)

    num_rings = config['Lamina']['rings']
    radius = config['Lamina']['radius']
    eulerangles = config['Lamina']['eulerangles']

    manager = core.Manager()

    with Timer('instantiation of lamina'):
        transform = AlbersProjectionMap(radius,
                                        eulerangles[0:3]).invmap
        hexagon = hx.HexagonArray(num_rings=num_rings, radius=radius,
                                  transform = transform)

        lamina = lam.LaminaArray(hexagon, config)
        add_lamina_LPU(config, 0, lamina, manager)

    start_simulation(config, manager)


if __name__ == '__main__':
    main()
