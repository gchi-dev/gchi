# =================
# !! THRESHOLDS DICTIONARY !!
# =================
hazard_thresholds = {
    "AT": [28, 32, 35, 40],  # °C
    "HI": [27, 32, 41, 54],  # °C
    "Hu": [30, 40, 45, 54],  # °C
    "WBT": [27.4, 28.9, 30.3, 35],  # °C
    "WBGT": [29, 30.5, 32, 37],  # °C
    "UTCIhot": [26, 32, 38, 46],  # °C
    "HWF": [0.052, 0.077, 0.110, 0.173],  # fraction of year
    "TR": [0.312, 0.532, 0.918, 0.997],  # fraction of year
    "TXC": [30, 35, 40, 45],  # °C
    "UTCIcold": [0, -13, -27, -40],  # °C
    "TNXp": [10, 5, 2, 0.5],  # unit percentile
    "FI": [0.852, 1.125, 1.579, 2.418], # FI index, unitless 
    "FWI": [12.23, 22.95, 36.83, 50],  # index
    "HDW": [3.932, 6.251, 9.746, 15.124],  # HDW index, unitless 
    "O3mon": [60, 65, 70, 100],  # ug/m^3
    "O3day": [100, 110, 120, 160],  # ug/m^3
    "PM2pt5mon": [5, 15, 25, 35],  # ug/m^3
    "PM2pt5day": [15, 37.5, 50, 75],  # ug/m^3
    "CDD": [0.805, 0.893, 0.964, 0.997], # fraction of year 
    "VSmalaria": [0, 0.333, 0.583, 0.833],  # fraction of year: marginal - endemic
    "VSzika": [0, 0.333, 0.583, 0.833],  # fraction of year: marginal - endemic
    "VSdengueAeg": [0, 0.333, 0.583, 0.833],  # fraction of year: marginal - endemic
    "VSdengueAlb": [0, 0.333, 0.583, 0.833],  # fraction of year: marginal - endemic
    "PR1day": [90, 95, 98, 99.5], # percentile
    "PR5day": [90, 95, 98, 99.5], # percentile
    "PRXmm": [20, 30, 40, 50], # mm/day
    "SPI": [-0.8, -1.3, -1.6, -2],  # index unitless
    "SPEI": [-0.8, -1.3, -1.6, -2],  # index unitless
    "VbrS": [0, 0.083, 0.167, 0.417],  # actual percentiles where 0, 0, 0.167, 0.417, but to differentiate L1 and L2 1 month is chosen for L2 (approx 97.2th percentile)
}

# thresholds from Table 1 Kudlackova et al 2025
# https://iopscience.iop.org/article/10.1088/1748-9326/ad97cf#erlad97cffA1
fwi_thresholds = {
    'D': [5.06, 10.9, 16.83, 22.45],
    'E': [7.59, 15.75, 24.01, 30.98],
    'F': [6.65, 14.1, 24.08, 36.7],
    'G': [8.97, 18.38, 31.41, 46.85],
    'H': [9.95, 20.73, 35.1, 51.26],
    'I': [20.73, 33.82, 48.65, 61.98],
    'J': [13.73, 23.33, 36.7, 52.44],
    'K': [18.1, 28.04, 40.82, 55.14],
    'L': [20.86, 32.19, 46.13, 61.99],
    'M': [5.43, 11.41, 21.83, 37.91],
    'N': [10.28, 20.44, 35.18, 54.96],
    'O': [10.28, 20.44, 35.18, 54.96],  # copied from N (no thresholds in Kudlackova, most will be masked by fuel mask)
    'P': [5.95, 12.61, 30.85, 62.0],    # copied from Q (no thresholds in Kudlackova, most will be masked by fuel mask)
    'Q': [5.95, 12.61, 30.85, 62.0],
    'R': [6.57, 12.86, 23.64, 39.84],
}
