from __future__ import division

import collections
import importlib

import networkx as nx
import numpy as np
PI = np.pi


def divceil(x, y):
    return (x+y-1)//y


class AmUtils(object):
    inds = collections.OrderedDict(
            [('a{}'.format(i+1), i+1) for i in range(6)])
    am_names = inds.keys()

    @classmethod
    def name_to_ind(cls, name):
        try:
            return cls.inds[name]
        except KeyError:
            print('"{}" is not a valid alpha process name'.format(name))
            raise

    @classmethod
    def ind_to_name(cls, index):
        try:
            return cls.am_names[index-1]
        except KeyError:
            print('"{}" is not a valid alpha process index'.format(index))
            raise

    @staticmethod
    def am_name(am_num):
        return 'Am{}'.format(am_num)

    @staticmethod
    def get_ampos_from_cart(i, lamina):
        cartid = i % len(lamina.cartridges)
        cartridge = lamina.cartridges[cartid]
        return cartridge.sphere_pos


class Cartridge(object):
    def __init__(self, element):
        '''
            element: ArrayElement object
        '''
        elev, azim = element.dima, element.dimb
        self._elev = elev
        self._azim = azim
        self.element = element

        self.neurons = collections.OrderedDict()
        self.synapses = []

    def add_neuron(self, neuron):
        if neuron.name not in self.neurons:
            self.neurons[neuron.name] = neuron

    def replace_am(self, name, am):
        ''' make alpha process 'name' a dummy neuron and
            connect it to its parent amacrine cell 'am'
        '''
        self.neurons[name].make_dummy()
        self.neurons[name].parent = am


    def get_neighborid(self, neighbor_dr):
        return self.element.get_neighborid(neighbor_dr)

    def is_neighbor_dummy(self, neighbor_num):
        return self.element.neighbors[neighbor_num].is_dummy

    @property
    def is_dummy(self):
        return self.element.is_dummy

    @property
    def sphere_pos(self):
        return self._elev, self._azim

    @property
    def gid(self):
        return self.element.gid


class NonColumn(object):
    def __init__(self, gid, elev, azim, params):
        self._gid = gid
        self._elev = elev
        self._azim = azim
        self.neuron = Neuron(params)

    @property
    def sphere_pos(self):
        return self._elev, self._azim

    @property
    def gid(self):
        return self._gid

    @property
    def selector(self):
        # assumes there is only one type of non columnar neuron
        # which is the amacrine cell
        return '/lam/am/{}'.format(self._gid)


class LaminaArray(object):
    def __init__(self, hex_array, config):
        self.hex_array = hex_array

        modelname = config['Lamina']['model']
        try:
            self.model = importlib.import_module('vision_models.{}'
                                                 .format(modelname))
        except ImportError:
            self.model = importlib.import_module('lamina.vision_models.{}'
                                                 .format(modelname))

        self._set_elements()
        self._update_neurons()

        relative_am = config['Lamina']['relative_am']
        if relative_am == 'half':
            self.n_amacrine = hex_array.num_elements//2
        elif relative_am == 'equal':
            self.n_amacrine = hex_array.num_elements
        else:
            self.n_amacrine = config['Lamina']['n_amacrine']

        # Have at least one Am cell
        self.n_amacrine = max(1, self.n_amacrine)

        self._connect_composition_II()
        self._connect_composition_I(config)

        self._generate_graph()

    def _set_elements(self):
        self._cartridges = [Cartridge(el) for el in self.hex_array.elements]

    def _update_neurons(self):
        for cart in self._cartridges:
            for neuron_params in self.model.CARTRIDGE_NEURON_LIST:
                neuron = CartridgeNeuron(cart, neuron_params)
                cart.add_neuron(neuron)

            for neuron_params in self.model.CARTRIDGE_IN_NEURON_LIST:
                neuron = CartridgeNeuron(cart, neuron_params, is_input=True)
                cart.add_neuron(neuron)

    def _connect_composition_I(self, config):
        am_params = self.model.AM_PARAMS
        n_amacrine = self.n_amacrine

        am_dic = collections.OrderedDict()

        for i in range(n_amacrine):
            am_elev, am_azim = AmUtils.get_ampos_from_cart(i, self)
            am_dic.update(
                {AmUtils.am_name(i): self.get_am(i, am_elev, am_azim,
                                                 am_params)})

        for cartridge in self.cartridges:
            for name in AmUtils.am_names:
                am_num = cartridge.gid % n_amacrine
                cartridge.replace_am(name, am_dic[AmUtils.am_name(am_num)])

        self._amacrines = am_dic

    def _connect_composition_II(self):
        model = self.model
        synapse_list = []
        for cartridge in self._cartridges:
            for synapse_dict in model.CARTRIDGE_CR_II_SYNAPSE_LIST:
                neighbor_num = synapse_dict['cart']
                if not cartridge.is_neighbor_dummy(neighbor_num):
                    synapse = Synapse(synapse_dict)
                    neighbor_id = cartridge.get_neighborid([neighbor_num])
                    synapse.link(
                        cartridge.neurons[synapse_dict['prename']],
                        self._cartridges[neighbor_id]
                            .neurons[synapse_dict['postname']])
                    synapse_list.append(synapse)

            for synapse_dict in model.INTRA_CARTRIDGE_SYNAPSE_LIST:
                synapse = Synapse(synapse_dict)
                synapse.link(
                    cartridge.neurons[synapse_dict['prename']],
                    cartridge.neurons[synapse_dict['postname']])
                synapse_list.append(synapse)

    def _generate_graph(self):
        G = nx.MultiDiGraph()
        num = 0

        # export neurons, each neuron has a unique id (num)
        for i, cartridge in enumerate(self._cartridges):
            # cartridge neurons e.g R1, L1,...
            # added once for each column
            # their order is the one they appear in
            # lamina model file
            # dummy neurons are not included
            for name, neuron in cartridge.neurons.items():
                if not neuron.is_dummy:
                    neuron.add_num(num)
                    G.add_node(num, neuron.params)
                    G.node[num].update({'selector':
                                        self.get_selector(i, name)})
                    num += 1

        for i, am in enumerate(self._amacrines.itervalues()):
            neuron = am.neuron
            neuron.add_num(num)
            G.add_node(num, neuron.params)
            G.node[num].update({'selector': am.selector})
            num += 1

        self._num_neurons = num

        num = 0
        # export synapses
        # `num` is not required since a synapse is identified by the neurons it
        # connects but it is convenient to have one
        for cartridge in self._cartridges:
            for neuron in cartridge.neurons.itervalues():
                for synapse in neuron.outgoing_synapses:
                    if not synapse.post_neuron.is_input:
                        synapse.params.update({'id': num})
                        synapse.process_before_export()
                        G.add_edge(synapse.pre_neuron.num,
                                   synapse.post_neuron.num,
                                   attr_dict=synapse.params)
                        num += 1

        self.G = G

    def get_graph(self):
        return self.G

    def get_am(self, i, am_elev, am_azim, params):
        return NonColumn(i, am_elev, am_azim, params)

    # Neuron representation
    def get_neurons(self, sublpu, sublpu_num):
        ommatidia = self._ommatidia
        ommatidia_num = len(ommatidia)
        start = divceil(sublpu*ommatidia_num, sublpu_num)
        end = divceil((sublpu+1)*ommatidia_num, sublpu_num)
        neurons = []
        # implicit assumption i is the same as
        # the global id of
        for i in range(start, end):
            ommatidium = ommatidia[i]
            for omm_neuron in ommatidium:
                neurons.append((ommatidium.gid,
                                omm_neuron.name,
                                ommatidium.sphere_pos))
        return neurons

    @property
    def cartridges(self):
        # assumes that cartridges were assigned to `_cartridges`
        # list in order of gid
        return self._cartridges

    @property
    def num_neurons(self):
        return self._num_neurons

    # Selector representation
    def get_all_selectors(self):
        selectors = []
        for i, cartridge in enumerate(self._cartridges):
            for name, neuron in cartridge.neurons.items():
                if not neuron.is_dummy:
                    selectors.append(self.get_selector(i, name))
        for i, am in enumerate(self._amacrines.itervalues()):
            selectors.append(am.selector)
        return selectors

    # method is not trivial since it handles the case
    # of dummy neuron
    def get_selector(self, cartid, name):
        neuron = self._cartridges[cartid].neurons[name]
        if neuron.is_dummy:
            return neuron.selector
        else:
            return '/lam/{}/{}'.format(cartid, name)

    # A convenient representation of all neuron information
    def get_neuron_objects(self):
        return self._cartridges, self._amacrines

    def index(self, cartid, name):
        return self._cartridges[cartid].neurons[name].num


class Neuron(object):
    def __init__(self, params):
        self._name = params.get('name')

        self._params = params.copy()

        self.outgoing_synapses = []
        self.incoming_synapses = []

    @property
    def params(self):
        return self._params

    @property
    def name(self):
        return self._name

    def add_outgoing_synapse(self, synapse):
        self.outgoing_synapses.append(synapse)

    def add_incoming_synapse(self, synapse):
        self.incoming_synapses.append(synapse)

    def remove_outgoing_synapse(self, synapse):
        self.outgoing_synapses.remove(synapse)

    def remove_incoming_synapse(self, synapse):
        self.incoming_synapses.remove(synapse)

    def __repr__(self):
        return 'neuron '+self.params['name']+': '+str(self.params)

    def __str__(self):
        return 'neuron '+str(self.params['name'])

    def add_num(self, num):
        self._num = num

    @property
    def num(self):
        return self._num


class CartridgeNeuron(Neuron):
    def __init__(self, cartridge, params, is_input=False):
        self.parent = cartridge
        self._input = is_input
        self._dummy = False

        super(CartridgeNeuron, self).__init__(params)

    @property
    def pos(self):
        return self.parent.element.pos

    # Notice it's different than is_dummy of `Cartridge` method
    @property
    def is_dummy(self):
        return self._dummy

    @property
    def is_input(self):
        return self._input

    @property
    def num(self):
        if self.is_dummy:
            # if .num is used it is considered that num
            # is called with 1 argument (not sure why)
            return self.parent.neuron._num
        else:
            # could avoid underscore but it gets complicated
            return self._num

    @property
    def selector(self):
        return self.parent.selector

    def make_dummy(self):
        self._dummy = True


class Synapse(object):

    def __init__(self, params):
        """ params: a dictionary of neuron parameters.
                    It can be passed as an attribute dictionary parameter
                    for a node in networkx library.
        """
        self._params = params.copy()

    def link(self, pre_neuron, post_neuron):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.pre_neuron.add_outgoing_synapse(self)
        self.post_neuron.add_incoming_synapse(self)

    def __repr__(self):
        return 'synapse from {} to {}: {}'.format(self.params['prename'],
                                                  self.params['postname'],
                                                  self.params)

    def __str__(self):
        return 'synapse from {} to {}'.format(self.params['prename'],
                                              self.params['postname'])

    @property
    def prenum(self):
        return self._prenum

    @prenum.setter
    def prenum(self, value):
        self._prenum = value

    @property
    def postnum(self):
        return self._postnum

    @postnum.setter
    def postnum(self, value):
        self._postnum = value

    @property
    def params(self):
        return self._params

    def process_before_export(self):
        # TODO remove what is not needed
        # merge with retina classes if exactly the same
        # assumes all conductances are gpot to gpot
        self._params.update({'class': 3})
        self._params.update({'conductance': True})
        if 'cart' in self._params.keys():
            del self._params['cart']
        if 'scale' in self.params.keys():
            self._params['slope'] *= self._params['scale']
            self._params['saturation'] *= self._params['scale']
            del self._params['scale']

def main():
    pass

if __name__ == "__main__":
    main()
