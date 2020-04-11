params = dict()

# El Euch, Rosenbaum (2016)
params['ER16'] = {
    'ALPHA': 0.75,
    'LAMBDA': 1,
    'RHO': 0.1,
    'XI': 1,
    'V0': 0.04,
    'VBAR': 0.2,
    'SPOT': 1,
    'RATE': 0,
}

# Gerhold, Gerstenecker, Pinter (2018)
params['GGP18'] = {
    'ALPHA': 0.6,
    'LAMBDA': 2,
    'RHO': -0.8,
    'XI': 0.2,
    'V0': 0.25,
    'VBAR': 0.2,
    'SPOT': 1,
    'RATE': 0,
}

# Gatheral, Radoicic (2019)
params['GR19'] = {
    'ALPHA': 0.55,
    'LAMBDA': 2,
    'RHO': -0.65,
    'XI': 0.4,
    'V0': 0.4,
    'VBAR': 0.04,
}

# FGGS18
params['FGGS18'] = {
    'ALPHA': 1,
    'V0': 0.0654,
    'VBAR': 0.0707,
    'LAMBDA': 0.6067,
    'XI': 0.2928,
    'RHO': -0.7571,
    'SPOT': 1,
    'RATE': 0,
}

# https://www.mathworks.com/help/fininst/optbyhestonni.html#d117e211346
params['HESTON'] = {
    'ALPHA': 1,
    'RATE': 0.03,
    'V0': 0.04,
    'VBAR': 0.05,
    'LAMBDA': 1,
    'XI': 0.2,
    'RHO': -0.7,
    'SPOT': 80,
}

# https://www.mathworks.com/matlabcentral/fileexchange/25771-heston-option-pricer
params['HESTON2'] = {
    'ALPHA': 1,
    'RATE': 0.02,
    'V0': 0.04,
    'VBAR': 0.06,
    'LAMBDA': 1.5,
    'XI': 0.7,
    'RHO': 0,
    'SPOT': 100,
}

# custom
params['HESTON3'] = {
    'ALPHA': 1,
    'RATE': 0,
    'V0': 0.4,
    'VBAR': 0.6,
    'LAMBDA': 1,
    'XI': 0.2,
    'RHO': -0.1,
    'SPOT': 1,
}

# custom
params['CUSTOM'] = {
    'ALPHA': 0.6,
    'LAMBDA': 1,
    'RHO': 0.1,
    'XI': 0.8,
    'V0': 0.5,
    'VBAR': 0.5,
    'SPOT': 1,
    'RATE': 0,
}

# # custom
# params['CUSTOM'] = {
#     'ALPHA': 0.6,
#     'LAMBDA': 1,
#     'RHO': 0.1,
#     'XI': 1,
#     'V0': 0.04,
#     'VBAR': 0.2,
#     'SPOT': 1,
#     'RATE': 0,
# }
