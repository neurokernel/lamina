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
        neuron = am.neuron
        synapses_to_remove = []
        for synapse in self.neurons[name].outgoing_synapses:
            synapse.params.update({'via': name+'_'+str(self.gid)})
            synapse.change_pre_neuron(neuron)
            neuron.add_outgoing_synapse(synapse)
            synapses_to_remove.append(synapse)
        
        for synapse in synapses_to_remove:
            self.neurons[name].remove_outgoing_synapse(synapse)
        
        synapses_to_remove = []
        
        for synapse in self.neurons[name].incoming_synapses:
            synapse.params.update({'via': name+'_'+str(self.gid)})
            synapse.change_post_neuron(neuron)
            neuron.add_incoming_synapse(synapse)
            synapses_to_remove.append(synapse)
        
        for synapse in synapses_to_remove:
            self.neurons[name].remove_incoming_synapse(synapse)
        
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
    def __init__(self, gid, elev, azim, params, is_input = False):
        self._gid = gid
        self._elev = elev
        self._azim = azim
        self.neuron = Neuron(params, is_input)
    
        self.outgoing_synapses = []
        self.incoming_synapses = []

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
            self.n_amacrine = config['Lamina']['number_am']

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

#    def _connect_composition_I(self, config):
#        am_params = self.model.AM_PARAMS
#        n_amacrine = self.n_amacrine
#
#        am_dic = collections.OrderedDict()
#
#        for i in range(n_amacrine):
#            am_elev, am_azim = AmUtils.get_ampos_from_cart(i, self)
#            am_dic.update(
#                {AmUtils.am_name(i): self.get_am(i, am_elev, am_azim,
#                                                 am_params)})
#
#        for cartridge in self.cartridges:
#            for name in AmUtils.am_names:
#                am_num = cartridge.gid % n_amacrine
#                cartridge.replace_am(name, am_dic[AmUtils.am_name(am_num)])
#
#        self._amacrines = am_dic

    def _connect_composition_Ia(self, config):
    # bound for amacrine connection distance
        bound = config['Composition']['Original']['b_amacrine']
        model = self.model
        n_amacrine = self.n_amacrine

        am_dic = collections.OrderedDict()
        synapse_list = []

#        angle = 2*np.pi*np.random.random(n_amacrine);
#        rad = self.hex_array._radius*np.sqrt(np.random.random(n_amacrine));
#        am_xpos = rad*np.cos(angle);
#        am_ypos = rad*np.sin(angle);

        angle = np.zeros(n_amacrine)
        rad = np.zeros(n_amacrine)
        am_xpos = np.zeros(n_amacrine)
        am_ypos = np.zeros(n_amacrine)
        
        for i in range(n_amacrine):
            dist = 0
            while dist < bound/2:
                angle_tmp = 2*np.pi*np.random.random()
                rad_tmp = self.hex_array._radius*np.sqrt(np.random.random())
                xpos_tmp = rad_tmp*np.cos(angle_tmp)
                ypos_tmp = rad_tmp*np.sin(angle_tmp)
                if i > 1:
                    dist = np.sqrt((xpos_tmp-am_xpos[:i])**2+(ypos_tmp-am_ypos[:i])**2).min()
                else:
                    dist = bound
            
            angle[i] = angle_tmp
            rad[i] = rad_tmp
            am_xpos[i] = xpos_tmp
            am_ypos[i] = ypos_tmp
        
        dim_a, dim_b = self.hex_array._transform(am_xpos, am_ypos)

        for i in range(n_amacrine):
            # XXX not the best way to access InterCartridgeNeuron
            am_dic.update(
                {'Am'+str(i): NonColumn(
                    i, dim_a[i], dim_b[i], model.AM_PARAMS, is_input=False)})
        # TODO: where to put synapse from Am to Am?
        # right now all synapse are listed inside a cartridge
        # even if the replaced dummy neuron results in Am to Am synapse
        alpha_profiles = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']
        fill = np.zeros((n_amacrine, len(self._cartridges)), np.int32);
        count = 0
        
        hex_loc = self.hex_array.hex_loc
        for i, cartridge in enumerate(self._cartridges):
            hx_loc = hex_loc[i]
            xpos = hex_loc[i,0]
            ypos = hex_loc[i,1]

            #calculate distance and find amacrine cells within
            #distance defined by bound
            dist = np.sqrt((xpos-am_xpos)**2 + (ypos-am_ypos)**2)
            suitable_am = np.nonzero(dist <= bound)[0]
            # if less than 4 neurons in the bound, get
            # the 4 closest amacrine cells
            if suitable_am.size < 4:
                suitable_am = np.argsort(dist)[0:4]

            for name in alpha_profiles:
                assigned = False
                for am_num in np.random.permutation(suitable_am):
                    if fill[am_num, count] < 3:
                        fill[am_num, count] += 1
                        synapses = cartridge.replace_am(name,
                                                        am_dic['Am'+str(am_num)])
                        #synapse_list.extend(synapses)
                        assigned = True
                        break
                if not assigned:
                    # causes problems later
                    raise ValueError('Likely need more amacrine cells.'
                                     ' {} in cartridge {} not assigned.'
                                     .format(name, cartridge.gid))
            count += 1

        self.am_xpos = am_xpos
        self.am_ypos = am_ypos
        self._amacrines = am_dic
        self.fill = fill
        #lamina.composition_rules.append( {'neurons': am_list, 'synapses': synapse_list} )

    def _connect_composition_I(self, config):
        np.random.seed(1000)
        bound = config['Composition']['Original']['b_amacrine']
        model = self.model
        n_amacrine = self.n_amacrine
        max_radius = self.hex_array.get_maximum_radius()

        am_dic = collections.OrderedDict()
        synapse_list = []
        
        minimum_space = 0.1 * max_radius
        
        am_xpos = np.empty(n_amacrine, dtype = np.double)
        am_ypos = np.empty(n_amacrine, dtype = np.double)
        
        for i in range(n_amacrine):
            if i != 0:
                while(1):
                    angle = 2*np.pi*np.random.random();
                    rad = np.sqrt(np.random.random())*max_radius;
                    x = rad*np.cos(angle);
                    y = rad*np.sin(angle);
                    if (np.sqrt( ((am_xpos[:i]-x)**2 + (am_ypos[:i]-y)**2).min()) >
                            ((1+np.random.randn()*0.1) * minimum_space)):
                        am_xpos[i] = x
                        am_ypos[i] = y
                        break
            else:
                angle = 2*np.pi*np.random.random();
                rad = np.sqrt(np.random.random())*max_radius;
                x = rad*np.cos(angle);
                y = rad*np.sin(angle);
                am_xpos[i] = x
                am_ypos[i] = y
    
        
        dim_a, dim_b = self.hex_array._transform(am_xpos, am_ypos)

        for i in range(n_amacrine):
            # XXX not the best way to access InterCartridgeNeuron
            am_dic.update(
                {'Am'+str(i): NonColumn(
                    i, dim_a[i], dim_b[i], model.AM_PARAMS)})

        # TODO: where to put synapse from Am to Am?
        # right now all synapse are listed inside a cartridge
        # even if the replaced dummy neuron results in Am to Am synapse
        alpha_profiles = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']
        fill = np.zeros((n_amacrine, len(self._cartridges)), np.int32);

        xx = self.hex_array.hex_loc[:,0] #np.asarray([a.pos[0] for a in lamina.hex_array], dtype = np.double)
        yy = self.hex_array.hex_loc[:,1] #np.asarray([a.pos[1] for a in lamina.elements], dtype = np.double)

        for i in range(n_amacrine):
            xpos = am_xpos[i]
            ypos = am_ypos[i]

            dist = np.sqrt((xpos-xx)**2 + (ypos-yy)**2)
            ind = np.argsort(dist)
            distsort = dist[ind]
            n = np.nonzero(distsort > self.hex_array._get_column_d()*2)[0][0]
            ind = ind[:n]

            fill[i, ind[0]] = 3
            for j in range(1, min(3, ind.size)):
                fill[i, ind[j]] = 2

            for j in range(3, min(12, ind.size)):
                fill[i, ind[j]] = 1
                
        sumfill = np.sum(fill, axis = 0)

        for i in range(len(self._cartridges)):
            xpos = xx[i]
            ypos = yy[i]

            if sumfill[i] < 6:
                dist = np.sqrt((xpos-am_xpos)**2 + (ypos-am_ypos)**2)
                ind = np.argsort(dist)
                less = 6 - sumfill[i]
                for j in range(less):
                    fill[ind[j], i] += 1
            elif sumfill[i] > 6:
                ind = np.nonzero(fill[:,i])[0]
                dist = np.sqrt( (xpos - am_xpos[ind])**2 + (ypos-am_ypos[ind])**2)
                ind1 = np.argsort(dist)
                more = sumfill[i] - 6
                for j in range(more):
                    fill[ind[ind1[j]], i] -= 1
        
        sumfill = np.sum(fill, axis = 0)
        assert(np.all(sumfill == 6))
        
        for i, cartridge in enumerate(self._cartridges):
            ind = np.nonzero(fill[:,i])[0]
            n_profiles = fill[ind, i]
        
            profiles = np.random.permutation(6)
            count = 0
            for j in range(ind.size):
                am = am_dic['Am'+str(ind[j])]
                for k in range(n_profiles[j]):
                    synapses = cartridge.replace_am(
                                    alpha_profiles[profiles[count]], am)
                    count += 1

        self.am_xpos = am_xpos
        self.am_ypos = am_ypos
        self.fill = fill
        self._amacrines = am_dic
        #lamina.composition_rules.append( {'neurons': am_list, 'synapses': synapse_list} )

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
    
    def generate_neuroarch_gexf(self):
        G_neuroarch = nx.MultiDiGraph()
        hex_loc = self.hex_array.hex_loc
    
        for i, cartridge in enumerate(self._cartridges):
            sphere_pos = cartridge.sphere_pos
            hx_loc = hex_loc[i]
            circuit_name = 'cartridge_{}'.format(i)

            G_neuroarch.add_node('circuit_'+circuit_name,
                                 {'name': circuit_name,
                                  '3d_elev': float(sphere_pos[0]),
                                  '3d_azim': float(sphere_pos[1]),
                                  '2d_x': float(hx_loc[0]),
                                  '2d_y': float(hx_loc[1])})

            for name, neuron in cartridge.neurons.items():
                neuron.circuit = circuit_name
                if not neuron.is_dummy:
                    neuron.id = 'neuron_{}_{}'.format(name, i)
                    if neuron.is_input:
                        neuron.id += '_port'
                    G_neuroarch.add_node(neuron.id, neuron.params.copy())
                    G_neuroarch.node[neuron.id].update(
                        {'name': name,
                         '3d_elev': float(sphere_pos[0]),
                         '3d_azim': float(sphere_pos[1]),
                         '2d_x': float(hx_loc[0]),
                         '2d_y': float(hx_loc[1]),
                         'circuit': circuit_name})
                    if neuron.is_input:
                        G_neuroarch.node[neuron.id].update(
                            {'selector': self.get_selector(i, name)})
                    
                    else: # assuming all other columnar neurons are output neurons
                        G_neuroarch.add_node(neuron.id+'_port',
                            {'class': 'Port', 'name': name+'_port',
                             'port_type': 'gpot', 'port_io': 'out',
                             'circuit': circuit_name,
                             'selector': self.get_selector(i, name)})
                        G_neuroarch.add_edge(neuron.id, neuron.id+'_port', type='directed')
                else:
                    neuron.id = 'dummy_{}_{}'.format(name, i)
                    G_neuroarch.add_node(neuron.id, neuron.params.copy())
                    G_neuroarch.node[neuron.id].update(
                        {'name': name,
                         '3d_elev': float(sphere_pos[0]),
                         '3d_azim': float(sphere_pos[1]),
                         '2d_x': float(hx_loc[0]),
                         '2d_y': float(hx_loc[1]),
                         'circuit': circuit_name,
                         'parent': 'Am_{}'.format(neuron.parent.gid)})
    
        G_neuroarch.add_node('circuit_cr1',
                             {'name': 'cr1'})
        G_neuroarch.add_node('circuit_cr2',
                             {'name': 'cr2'})
    
        for i, am in enumerate(self._amacrines.itervalues()):
            neuron = am.neuron
            neuron.id = 'neuron_Am_{}'.format(i)
            G_neuroarch.add_node(neuron.id, neuron.params)
            G_neuroarch.node[neuron.id].update(
                    {'name': 'Am',
                     'circuit': 'cr1',
                     '3d_elev': float(am.sphere_pos[0]),
                     '3d_azim': float(am.sphere_pos[1]),
                     '2d_x': float(self.am_xpos[i]),
                     '2d_y': float(self.am_ypos[i])})
#            G_neuroarch.add_node(neuron.id+'_port',
#                        {'class': 'Port', 'name': neuron.id+'_port',
#                         'port_type': 'gpot', 'port_io': 'out',
#                         'circuit': 'cr1',
#                         'selector': am.selector})
#            G_neuroarch.add_edge(neuron.id, neuron.id+'_port', type='directed')

        num = 0
        for cartridge in self._cartridges:
            for neuron in cartridge.neurons.itervalues():
#                if not neuron.is_dummy:
                    for synapse in neuron.outgoing_synapses:
    #                    if not synapse.post_neuron.is_input:
                            synapse_id = 'synapse_{}'.format(num)
                            synapse.process_before_export()
                            G_neuroarch.add_node(synapse_id, synapse.params)
                            
                            if isinstance(synapse.pre_neuron, CartridgeNeuron) \
                             and isinstance(synapse.post_neuron, CartridgeNeuron):
                                if synapse.pre_neuron.parent == synapse.post_neuron.parent:
                                    G_neuroarch.node[synapse_id].update(
                                        {'circuit': synapse.pre_neuron.circuit})
                                else:
                                    G_neuroarch.node[synapse_id].update(
                                        {'circuit': 'cr2'})
                            else:
                                G_neuroarch.node[synapse_id].update(
                                        {'circuit': 'cr1'})
                            G_neuroarch.add_edge(synapse.pre_neuron.id, synapse_id,
                                       type = 'directed')
                            G_neuroarch.add_edge(synapse_id, synapse.post_neuron.id,
                                       type = 'directed')
                            num+=1
        
        for am in self._amacrines.itervalues():
            neuron = am.neuron
            for synapse in neuron.outgoing_synapses:
                synapse_id = 'synapse_{}'.format(num)
                synapse.process_before_export()
                G_neuroarch.add_node(synapse_id, synapse.params)
                G_neuroarch.node[synapse_id].update({'circuit': 'cr1'})
                G_neuroarch.add_edge(synapse.pre_neuron.id, synapse_id,
                                       type = 'directed')
                G_neuroarch.add_edge(synapse_id, synapse.post_neuron.id,
                           type = 'directed')
                num+=1

        return G_neuroarch

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
                neuron.id = 'lam_{}_{}'.format(name, i)
                if not neuron.is_dummy:
                    neuron.add_num(num)
                    tmp = neuron.params.copy()
                    try:
                        tmp.pop('genetic.neurotransmitter')
                    except:
                        pass
                    G.add_node(neuron.id, tmp)
                    if neuron.is_input:
                        G.node[neuron.id].update({'selector':
                            self.get_selector(i, name)})
                        G.add_node(neuron.id+'_aggregator',
                                   {'class': 'Aggregator',
                                    'name': name})
                        G.add_node(neuron.id+'_aggregator_port',
                                   {'class': 'Port',
                                    'port_type': 'gpot', 'port_io': 'out',
                                    'selector': self.get_selector(i, name)+'_agg',
                                    'name': name})
                        G.add_edge(neuron.id+'_aggregator',
                                   neuron.id+'_aggregator_port',
                                   variable = 'I')
                        G.add_edge(neuron.id,
                                   neuron.id+'_aggregator',
                                   variable = 'V')
                    else: # assuming all other columnar neurons are output neurons
                        G.add_node(neuron.id+'_port',
                            {'class': 'Port', 'name': name+'_port',
                             'port_type': 'gpot', 'port_io': 'out',
                             'selector': self.get_selector(i, name)})
                        G.add_edge(neuron.id, neuron.id+'_port')#, type='directed')
                    num += 1

        for i, am in enumerate(self._amacrines.itervalues()):
            neuron = am.neuron
            neuron.id = 'lam_am_{}'.format(i)
            neuron.add_num(num)
            tmp = neuron.params.copy()
            try:
                tmp.pop('genetic.neurotransmitter')
            except:
                pass
            G.add_node(neuron.id, tmp)
#            G.add_node(neuron.id+'_port',
#                        {'class': 'Port', 'name': name+'_port',
#                         'port_type': 'gpot', 'port_io': 'out',
#                         'selector': am.selector})
#            G.add_edge(neuron.id, neuron.id+'_port', type='directed')
            num += 1

        self._num_neurons = num

        num = 0
        # export synapses
        # `num` is not required since a synapse is identified by the neurons it
        # connects but it is convenient to have one
        
        # ignored all Am->Am connection
        a = 0
        for cartridge in self._cartridges:
            for neuron in cartridge.neurons.itervalues():
#                if not neuron.is_dummy:
                    for synapse in neuron.outgoing_synapses:
    #                    if not synapse.post_neuron.is_input:
                            synapse_id = 'syn_{}'.format(num)
                            synapse.process_before_export()
                            tmp = synapse.params.copy()
                            try:
                                tmp.pop('via')
                            except:
                                pass
            
                            G.add_node(synapse_id, tmp)
                            G.add_edge(synapse.pre_neuron.id, synapse_id)
                            if synapse.post_neuron.is_input:
                                G.add_edge(synapse_id,
                                           synapse.post_neuron.id+'_aggregator')
                            else:
                                G.add_edge(synapse_id, synapse.post_neuron.id)
                            num += 1
        
        for am in self._amacrines.itervalues():
            neuron = am.neuron
            for synapse in neuron.outgoing_synapses:
                synapse_id = 'synapse_{}'.format(num)
                synapse.process_before_export()
                tmp = synapse.params
                try:
                    tmp.pop('via')
                except:
                    pass
            
                G.add_node(synapse_id, tmp)
                G.add_edge(synapse.pre_neuron.id, synapse_id,
                           type = 'directed')
                if synapse.post_neuron.is_input:
                    G.add_edge(synapse_id,
                               synapse.post_neuron.id+'_aggregator')
                else:
                    G.add_edge(synapse_id, synapse.post_neuron.id)
                num+=1

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
                    if neuron.is_input:
                        selectors.append(self.get_selector(i, name)+'_agg')

#        for i, am in enumerate(self._amacrines.itervalues()):
#            selectors.append(am.selector)
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
    def __init__(self, params, is_input = False):
        self._name = params.get('name')
        self._input = is_input

        self._params = params.copy()

        self.outgoing_synapses = []
        self.incoming_synapses = []

    @property
    def params(self):
        return self._params

    @property
    def name(self):
        return self._name
    
    @property
    def is_input(self):
        return self._input

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
        self._dummy = False

        super(CartridgeNeuron, self).__init__(params, is_input)

    @property
    def pos(self):
        return self.parent.element.pos

    # Notice it's different than is_dummy of `Cartridge` method
    @property
    def is_dummy(self):
        return self._dummy

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
    
    def change_post_neuron(self, post_neuron):
        self.post_neuron = post_neuron
        self.params['postname'] = post_neuron.name
    
    def change_pre_neuron(self, pre_neuron):
        self.pre_neuron = pre_neuron
        self.params['prename'] = pre_neuron.name

    def process_before_export(self):
        # TODO remove what is not needed
        # merge with retina classes if exactly the same
        # assumes all conductances are gpot to gpot
        self._params.update({'conductance': True})
        if 'cart' in self._params.keys():
            del self._params['cart']
        if 'scale' in self.params.keys():
            self._params['slope'] *= self._params['scale']
            self._params['saturation'] *= self._params['scale']
            del self._params['scale']

def main():
    from retina.screen.map.mapimpl import AlbersProjectionMap
    import retina.geometry.hexagon as hex
    from retina.configreader import ConfigReader
    import lamina.lamina as lam
    import networkx as nx
    config=ConfigReader('retlam_default.cfg','../template_spec.cfg').conf
    transform = AlbersProjectionMap(config['Retina']['radius'],config['Retina']['eulerangles']).invmap
    hexarray = hex.HexagonArray(num_rings = 14, radius = config['Retina']['radius'], transform = transform)
    a = lam.LaminaArray(hexarray, config)
    G = a.generate_neuroarch_gexf()
    nx.write_gexf(G, 'lamina_neuroarch.gexf.gz')

if __name__ == "__main__":
    main()
