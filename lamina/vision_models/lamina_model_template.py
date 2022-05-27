# **** [vision_class_template.py]               ****
# **** Do not modify this file                  ****
# **** Copy into a new file and modify that one ****

# graded potential to graded potential synapses documented in
# Neurokernel RFC 2 http://neurokernel.github.io/rfc/nk-rfc2.pdf
# RFC   | Here
# ------------------
# k     | slope
# V_th  | threshold
# n     | power
# g_sat | saturation

R_THRESHOLD = -80.0
L1_THRESHOLD = -50.5
L2_THRESHOLD = -50.5
L3_THRESHOLD = -50.5
L4_THRESHOLD = -50.5
L5_THRESHOLD = -50.5
C2_THRESHOLD = -50.0
C3_THRESHOLD = -50.0
AM_THRESHOLD = -50.0
T1_THRESHOLD = -50.0

ACH_RP = 0.0
HIST_RP = -80.0
GLU_RP = 0.0
GABA_RP = -80.0

AM_RP = GLU_RP

SLOPE = 0.02
SATURATION = 0.0008

CARTRIDGE_IN_NEURON_LIST = [
    {
        'name': 'R{}'.format(i+1), 'class': 'Port',
        'port_type': 'gpot', 'port_io': 'in'
    }
    for i in range(6)
]

CARTRIDGE_NEURON_LIST = [
    {
        'name': 'L1', 'class': 'MorrisLecar',
        'V1': -1.0, 'V2': 15, 'V3': -5.0, 'V4': 10.0,
        'V_L': -50.0, 'V_Ca': 10.0, 'V_k': -70.0,
        'g_L': 0.5, 'G_Ca': 1.1, 'G_k': 2.0,
        'phi': 0.0025, 'initV': -50., 'initn': 0.5, 'offset': 0.02
    },
    {
        'name': 'L2', 'class': 'MorrisLecar',
        'V1': -1.0, 'V2': 15, 'V3': -5.0, 'V4': 10.0,
        'V_L': -50.0, 'V_Ca': 10.0, 'V_k': -70.0,
        'g_L': 0.5, 'G_Ca': 1.1, 'G_k': 2.0,
        'phi': 0.0025, 'initV': -50., 'initn': 0.5, 'offset': 0.02
    },
    {
        'name': 'L3', 'class': 'MorrisLecar',
        'V1': -1.0, 'V2': 15, 'V3': -5.0, 'V4': 10.0,
        'V_L': -50.0, 'V_Ca': 10.0, 'V_k': -70.0,
        'g_L': 0.5, 'G_Ca': 1.1, 'G_k': 2.0,
        'phi': 0.0025, 'initV': -50., 'initn': 0.5, 'offset': 0.02
    },
    {
        'name': 'L4', 'class': 'MorrisLecar',
        'V1': -1.0, 'V2': 15, 'V3': -5.0, 'V4': 10.0,
        'V_L': -50.0, 'V_Ca': 10.0, 'V_k': -70.0,
        'g_L': 0.5, 'G_Ca': 1.1, 'G_k': 2.0,
        'phi': 0.0025, 'initV': -50., 'initn': 0.5, 'offset': 0.02
    },
    {
        'name': 'L5', 'class': 'MorrisLecar',
        'V1': -1.0, 'V2': 15, 'V3': -5.0, 'V4': 10.0,
        'V_L': -50.0, 'V_Ca': 10.0, 'V_k': -70.0,
        'g_L': 0.5, 'G_Ca': 1.1, 'G_k': 2.0,
        'phi': 0.0025, 'initV': -50., 'initn': 0.5, 'offset': 0.02
    },
    {
        'name':'T1', 'class':'MorrisLecar',
        'V1': -1.0, 'V2': 15, 'V3': -5.0, 'V4': 10.0,
        'V_L': -50.0, 'V_Ca': 10.0, 'V_k': -70.0,
        'g_L': 0.5, 'G_Ca': 1.1, 'G_k': 2.0,
        'phi': 0.0025, 'initV': -50., 'initn': 0.5, 'offset': 0.02
    },
    {
        'name':'C2', 'class':'MorrisLecar',
        'V1': -1.0, 'V2': 15, 'V3': -5.0, 'V4': 10.0,
        'V_L': -50.0, 'V_Ca': 10.0, 'V_k': -70.0,
        'g_L': 0.5, 'G_Ca': 1.1, 'G_k': 2.0,
        'phi': 0.0025, 'initV': -50., 'initn': 0.5, 'offset': 0.02
    },
    {
        'name':'C3', 'class':'MorrisLecar',
        'V1': -1.0, 'V2': 15, 'V3': -5.0, 'V4': 10.0,
        'V_L': -50.0, 'V_Ca': 10.0, 'V_k': -70.0,
        'g_L': 0.5, 'G_Ca': 1.1, 'G_k': 2.0,
        'phi': 0.0025, 'initV': -50., 'initn': 0.5, 'offset': 0.02
    }
] + \
[
    {
        'name': 'a{}'.format(i+1)
    }
    for i in range(6)
]

AM_PARAMS = {
    'name': 'Am', 'class': 'MorrisLecar',
        'V1': -1.0, 'V2': 15, 'V3': 0.0, 'V4': 30.0,
        'V_L': -50.0, 'V_Ca': 10.0, 'V_k': -70.0,
        'g_L': 0.5, 'G_Ca': 1.1, 'G_k': 2.0,
        'phi': 0.2, 'initV': -51.84, 'initn': 0.0306, 'offset': 0.0
}


INTRA_CaRTRIDGE_SYNAPSE_LIST = [
    {'prename':'R1', 'postname':'L1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.0008,
    'scale':40, 'mode':0},
    {'prename':'R2', 'postname':'L1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.0008,
    'scale':43, 'mode':0},
    {'prename':'R3', 'postname':'L1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.0008,
    'scale':37, 'mode':0},
    {'prename':'R4', 'postname':'L1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.0008,
    'scale':38, 'mode':0},
    {'prename':'R5', 'postname':'L1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.0008,
    'scale':38, 'mode':0},
    {'prename':'R6', 'postname':'L1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.0008,
    'scale':45, 'mode':0},

    {'prename':'R1', 'postname':'L2', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.0008,
    'scale':46, 'mode':0},
    {'prename':'R2', 'postname':'L2', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.0008,
    'scale':45, 'mode':0},
    {'prename':'R3', 'postname':'L2', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.0008,
    'scale':39, 'mode':0},
    {'prename':'R4', 'postname':'L2', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.0008,
    'scale':41, 'mode':0},
    {'prename':'R5', 'postname':'L2', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.0008,
    'scale':39, 'mode':0},
    {'prename':'R6', 'postname':'L2', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.0008,
    'scale':47, 'mode':0},

    {'prename':'R1', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.0008,
    'scale':11, 'mode':0},
    {'prename':'R2', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.0008,
    'scale':10, 'mode':0},
    {'prename':'R3', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.0008,
    'scale':4, 'mode':0},
    {'prename':'R4', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.0008,
    'scale':8, 'mode':0},
    {'prename':'R5', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.0008,
    'scale':6, 'mode':0},
    {'prename':'R6', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00002, 'power':1.0, 'saturation':0.0008,
    'scale':12, 'mode':0},

    {'prename':'L2', 'postname':'L1', 'class':'PowerGPotGPot',#
    'cart':None, 'reverse':ACH_RP, 'delay':1,
    'threshold':L2_THRESHOLD, 'slope':0.001, 'power':1.0, 'saturation':0.02,
    'scale':3, 'mode':0},
    {'prename':'L2', 'postname':'L4', 'class':'PowerGPotGPot',#
    'cart':None, 'reverse':ACH_RP, 'delay':1,
    'threshold':L2_THRESHOLD, 'slope':0.002, 'power':1.0, 'saturation':0.03,
    'scale':4, 'mode':0},
    {'prename':'L2', 'postname':'L5', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':ACH_RP, 'delay':1,
    'threshold':L2_THRESHOLD, 'slope':0.002, 'power':1.0, 'saturation':0.01,
    'scale':1, 'mode':0},

    {'prename':'L4', 'postname':'L2', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':ACH_RP, 'delay':1,
    'threshold':L4_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.1,
    'scale':3, 'mode':0},

    {'prename':'R1', 'postname':'a1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00001, 'power':1.0, 'saturation':0.002,
    'scale':19, 'mode':0},
    {'prename':'R2', 'postname':'a1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00001, 'power':1.0, 'saturation':0.002,
    'scale':16, 'mode':0},
    {'prename':'R2', 'postname':'a2', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00001, 'power':1.0, 'saturation':0.002,
    'scale':22, 'mode':0},
    {'prename':'R3', 'postname':'a2', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00001, 'power':1.0, 'saturation':0.002,
    'scale':18, 'mode':0},
    {'prename':'R3', 'postname':'a3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00001, 'power':1.0, 'saturation':0.002,
    'scale':20, 'mode':0},
    {'prename':'R4', 'postname':'a3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00001, 'power':1.0, 'saturation':0.002,
    'scale':16, 'mode':0},
    {'prename':'R4', 'postname':'a4', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00001, 'power':1.0, 'saturation':0.002,
    'scale':17, 'mode':0},
    {'prename':'R5', 'postname':'a4', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00001, 'power':1.0, 'saturation':0.002,
    'scale':26, 'mode':0},
    {'prename':'R5', 'postname':'a5', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00001, 'power':1.0, 'saturation':0.002,
    'scale':10, 'mode':0},
    {'prename':'R6', 'postname':'a5', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00001, 'power':1.0, 'saturation':0.002,
    'scale':14, 'mode':0},
    {'prename':'R6', 'postname':'a6', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00001, 'power':1.0, 'saturation':0.002,
    'scale':22, 'mode':0},
    {'prename':'R1', 'postname':'a6', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':HIST_RP, 'delay':1,
    'threshold':R_THRESHOLD, 'slope':0.00001, 'power':1.0, 'saturation':0.002,
    'scale':17, 'mode':0},

    {'prename':'a1', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.01,
    'scale':1, 'mode':0},
    {'prename':'a2', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.01,
    'scale':1, 'mode':0},
    {'prename':'a3', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.01,
    'scale':1, 'mode':0},
    {'prename':'a4', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.01,
    'scale':3, 'mode':0},
    {'prename':'a5', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.01,
    'scale':5, 'mode':0},
    {'prename':'a6', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.01,
    'scale':1, 'mode':0},

    {'prename':'a5', 'postname':'L4', 'class':'PowerGPotGPot',#
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0003, 'power':1.0, 'saturation':0.06,
    'scale':3, 'mode':0},
    {'prename':'a4', 'postname':'L4', 'class':'PowerGPotGPot',#
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0003, 'power':1.0, 'saturation':0.06,
    'scale':1, 'mode':0},
    {'prename':'a4', 'postname':'L5', 'class':'PowerGPotGPot',#
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0003, 'power':1.0, 'saturation':0.06,
    'scale':4, 'mode':0},

    {'prename':'a1', 'postname':'T1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.06,
    'scale':8, 'mode':0},
    {'prename':'a2', 'postname':'T1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.06,
    'scale':6, 'mode':0},
    {'prename':'a3', 'postname':'T1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.06,
    'scale':7, 'mode':0},
    {'prename':'a4', 'postname':'T1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.06,
    'scale':12, 'mode':0},
    {'prename':'a5', 'postname':'T1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.06,
    'scale':3, 'mode':0},
    {'prename':'a6', 'postname':'T1', 'class':'PowerGPotGPot',
    'cart':None, 'reverse':AM_RP, 'delay':1,
    'threshold':AM_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.06,
    'scale':13, 'mode':0}
]

CARTRIDGE_CR_II_SYNAPSE_LIST = [
    {'prename':'L2', 'postname':'L4', 'class':'PowerGPotGPot',
    'cart':6, 'reverse':ACH_RP, 'delay':1,
    'threshold':L2_THRESHOLD, 'slope':0.002, 'power':1.0, 'saturation':0.03,
    'scale':4, 'mode':0},
    {'prename':'L2', 'postname':'L4', 'class':'PowerGPotGPot',
    'cart':5, 'reverse':ACH_RP, 'delay':1,
    'threshold':L2_THRESHOLD, 'slope':0.002, 'power':1.0, 'saturation':0.03,
    'scale':2, 'mode':0},

    {'prename':'L4', 'postname':'L2', 'class':'PowerGPotGPot',#
    'cart':3, 'reverse':ACH_RP, 'delay':1,
    'threshold':L4_THRESHOLD, 'slope':0.002, 'power':1.0, 'saturation':0.1,
    'scale':4, 'mode':0},
    {'prename':'L4', 'postname':'L5', 'class':'PowerGPotGPot',
    'cart':3, 'reverse':ACH_RP, 'delay':1,
    'threshold':L4_THRESHOLD, 'slope':0.001, 'power':1.0, 'saturation':0.5,
    'scale':1, 'mode':0},

    {'prename':'L4', 'postname':'L4', 'class':'PowerGPotGPot',
    'cart':4, 'reverse':ACH_RP, 'delay':1,
    'threshold':L4_THRESHOLD, 'slope':0.002, 'power':1.0, 'saturation':0.05,
    'scale':2, 'mode':0},

    {'prename':'L4', 'postname':'L2', 'class':'PowerGPotGPot',
    'cart':2, 'reverse':ACH_RP, 'delay':1,
    'threshold':L4_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.2,
    'scale':2, 'mode':0},
    {'prename':'L4', 'postname':'L3', 'class':'PowerGPotGPot',
    'cart':2, 'reverse':ACH_RP, 'delay':1,
    'threshold':L4_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.05,
    'scale':1, 'mode':0},
    {'prename':'L4', 'postname':'L4', 'class':'PowerGPotGPot',
    'cart':2, 'reverse':ACH_RP, 'delay':1,
    'threshold':L4_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.05,
    'scale':1, 'mode':0},
    {'prename':'L4', 'postname':'L5', 'class':'PowerGPotGPot',
    'cart':2, 'reverse':ACH_RP, 'delay':1,
    'threshold':L4_THRESHOLD, 'slope':0.0005, 'power':1.0, 'saturation':0.1,
    'scale':1, 'mode':0},
]

