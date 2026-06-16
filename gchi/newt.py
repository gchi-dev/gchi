import numpy as np
from numba import vectorize

# Application of Noniterative Noniterative Evaluation of Wet-bulb Temperature (NEWT) method (Rogers and Warren, 2023; https://essopenarchive.org/users/714325/articles/698601-fast-and-accurate-calculation-of-wet-bulb-temperature-for-humid-heat-extremes?commit=a888ca0a0d28f09b4b49826f987739fa5180ebec).
# # Portions of this file are adapted from rob-warren/atmos (Rob Warren) (https://github.com/AusClimateService/atmos)
# Licensed under CC-BY-4.0 — see LICENSES/CC-BY-4.0-atmos.txt

# ── Constants ────────────────────────────────────────────────────────────────
"""
Physical constants

References:
* Ambaum, M. H., 2020. Accurate, simple equation for saturated vapour pressure
    over water and ice. Quart. J. Roy. Met. Soc., 146, 4252-4258,
    https://doi.org/10.1002/qj.3899.
* Feistel, R., and W. Wagner, 2006: A New Equation of State for H2O Ice Ih.
    J. Phys. Chem. Ref. Data, 35, 2021-1047 https://doi.org/10.1063/1.2183324.
* Guildner, L. A., D. P. Johnson, and F. E. Jones, 1976: Vapor pressure of water
    at its triple point. J. Res. Natl. Bur. Stand., 80A, 505-521,
    https://doi.org/10.6028/jres.080A.054.
* Wagner, W., and A. Pruß, 2002: The IAPWS Formulation 1995 for the 
    Thermodynamic Properties of Ordinary Water Substance for General and
    Scientific Use. J. Phys. Chem. Ref. Data, 31, 387-535,
    https://doi.org/10.1063/1.1461829.

"""

# Acceleration due to gravity (m/s2)
g = 9.81
# Specific gas constant for dry air (J/kg/K)
Rd = 287.0
# Specific gas constant for water vapour (J/kg/K)
Rv = 461.5
# Ratio of gas constants for dry air and water vapour
eps = Rd / Rv
# Isobaric specific heat of dry air (J/kg/K)
cpd = 1005.0
# Isobaric specific heat of water vapour (J/kg/K)
cpv = 2040.0  # optimised value from Ambaum (2020)
# Isobaric specific heat of liquid water (J/kg/K)
cpl = 4220.0  # triple-point value from Wagner and Pruß (2002)
# Isobaric specific heat of ice (J/kg/K)
cpi = 2097.0  # triple-point value from Feistel and Wagner (2006)
# Reference pressure (Pa)
p_ref = 1.0e5
# Triple point temperature (K)
T0 = 273.16
# Saturation vapour pressure at the triple point (Pa)
es0 = 611.657  # Guildner et al. (1976)
# Latent heat of vaporisation at the triple point (J/kg)
Lv0 = 2.501e6  # Wagner and Pruß (2002)
# Latent heat of freezing at the triple point (J/kg)
Lf0 = 0.333e6  # Feistel and Wagner (2006)
# Latent heat of sublimation at the triple point (J/kg)
Ls0 = Lv0 + Lf0  
# Temperature above which all condensate is assumed to be liquid (K)
T_liq = 273.15  
# Temperature below which all condensate is assumed to be ice (K)
T_ice = 253.15

# ── pseudoadiabat (paste Document 2 functions here as-is) ────────────────────
@vectorize(['float32(float32, float32)', 'float64(float64, float64)'], nopython=True)
def wbpt(p, T):
    """
    Computes the wet-bulb potential temperature (WBPT) thw of the
    pseudoadiabat that passes through pressure p and temperature T.

    Uses polynomial approximations from Moisseeva and Stull (2017)
    with revised coefficients.

    Moisseeva, N. and Stull, R., 2017. A noniterative approach to
        modelling moist thermodynamics. Atmospheric Chemistry and
        Physics, 17, 15037-15043.

    Args:
        p: pressure (Pa)
        T: temperature (K)

    Returns:
        thw: wet-bulb potential temperature (K)

    """

    # Convert p to hPa and T to degC
    p_ = p / 100.
    T_ = T - 273.15

    # Compute theta-w using Eq. 4-6 from Moisseeva & Stull 2017
    if (T_ >= -100.) & (T_ <= 50.) & (p_ <= 1100.) & (p_ >= 50.):
        Tref = 5.480079558912395754e+01 + p_*(-5.702912372295836363e-01 + p_*(6.214635352625029474e-03 + p_*(-6.634002758531432769e-05 + p_*(4.775890354003946154e-07 + p_*(-2.284857526370373519e-09 + p_*(6.641586128297075239e-12 + p_*(-7.712521760947926640e-15 + p_*(-2.044872914238500011e-17 + p_*(1.120406966307735934e-19 + p_*(-2.393726252696363534e-22 + p_*(2.666858658235125377e-25 + p_*(-8.684116177147550627e-29 + p_*(-1.672712626829787198e-31 + p_*(2.183518619078883688e-34 + p_*(-1.547439304626812778e-38 + p_*(-1.937257327731052786e-40 + p_*(2.161580416088237645e-43 + p_*(-1.160157488827817865e-46 + p_*(3.277570207101446812e-50 + p_*(-3.923759467350554795e-54))))))))))))))))))))  # noqa: E501
        thw = 4.232924249278688222e+01 + T_*(5.718008668788681081e-01 + T_*(6.676735845163824824e-03 + T_*(4.022733506471462767e-05 + T_*(-9.509825840570524462e-07 + T_*(-7.879928387880090530e-09 + T_*(1.767656648682178749e-10 + T_*(2.337360533109320417e-12 + T_*(-3.564315751634256907e-14 + T_*(-6.615642909573364126e-16 + T_*(5.465028726086066100e-18 + T_*(1.710624575834384904e-19 + T_*(-1.714074684770886933e-22 + T_*(-3.374318285502554966e-23 + T_*(-1.906956286638301820e-25 + T_*(3.748889164551010026e-27 + T_*(4.895600997189238897e-29 + T_*(-3.555428293757981745e-32 + T_*(-3.897220799151443075e-33 + T_*(-2.551411607182840165e-35 + T_*(-5.417589606240612172e-38)))))))))))))))))))) + Tref*(3.623240553023549526e-01 + T_*(1.023375828905026411e-03 + T_*(7.652197539188903771e-05 + T_*(-3.335127874231546565e-07 + T_*(-1.156314790854086800e-08 + T_*(5.997724820856132307e-11 + T_*(3.205744785514147323e-12 + T_*(-3.725255127402225228e-15 + T_*(-4.985713597883638890e-16 + T_*(-1.788582915460586641e-17 + T_*(8.672108280207142462e-20 + T_*(1.195118892414961522e-20 + T_*(-3.811303263205360248e-24 + T_*(-3.991925996145037697e-24 + T_*(-1.495975110164947026e-26 + T_*(6.968870338282429091e-28 + T_*(6.209536671787076346e-30 + T_*(-3.566388507077176018e-32 + T_*(-7.450360218688953006e-34 + T_*(-3.816398138670827989e-36 + T_*(-6.712873080670899696e-39)))))))))))))))))))) + Tref*(2.901968635714725214e-03 + T_*(6.487857234220085253e-05 + T_*(7.743602693621120145e-07 + T_*(-9.204198773051246169e-09 + T_*(-2.122032402927605809e-10 + T_*(4.125625669666862445e-12 + T_*(3.532509087599244118e-14 + T_*(-5.242786376794922436e-16 + T_*(-7.260942673196442122e-18 + T_*(-2.453561164912172901e-19 + T_*(5.678598204071161723e-21 + T_*(1.229928951189520298e-22 + T_*(-2.566095852346387466e-24 + T_*(-2.594462117958716581e-26 + T_*(6.198016439919091221e-28 + T_*(4.079274536668629507e-30 + T_*(-8.523972978819286856e-32 + T_*(-6.333007168641729819e-34 + T_*(4.884637765078024090e-36 + T_*(5.970619077275256973e-38 + T_*(1.583552627627211185e-40)))))))))))))))))))) + Tref*(6.709096824546971884e-05 + T_*(1.164430354507516326e-06 + T_*(1.492519185739810694e-08 + T_*(-1.004992535578806199e-10 + T_*(-5.194638284127568879e-12 + T_*(-8.852593174458557238e-14 + T_*(7.743098463983663378e-15 + T_*(-1.349430816115996717e-17 + T_*(-5.111520085301861189e-18 + T_*(7.246506699900768993e-20 + T_*(1.688571447195582101e-21 + T_*(-4.458601255074950933e-23 + T_*(-3.906571883790175152e-25 + T_*(1.401012855147709264e-26 + T_*(1.049655774799796372e-28 + T_*(-2.326684433917799696e-30 + T_*(-2.546715297351952591e-32 + T_*(1.076294752796537571e-34 + T_*(2.697444862568323708e-36 + T_*(1.404413350272535558e-38 + T_*(2.444355701979738372e-41)))))))))))))))))))) + Tref*(5.609138211497678676e-07 + T_*(2.619248932484586790e-08 + T_*(1.268762007103502264e-10 + T_*(-1.010471605550804941e-12 + T_*(-1.709919997846759906e-13 + T_*(-9.956527977396212394e-16 + T_*(2.437200472916002895e-16 + T_*(-2.053684743415526158e-18 + T_*(-1.347065496266485984e-19 + T_*(2.110916506052379395e-21 + T_*(3.809682178285610452e-23 + T_*(-7.906486212697025412e-25 + T_*(-6.716781189658700860e-27 + T_*(1.545503563949260188e-28 + T_*(8.638277498899926506e-31 + T_*(-1.671970193463004779e-32 + T_*(-7.800495852299937013e-35 + T_*(9.773608301957742171e-37 + T_*(3.864485572169008074e-39 + T_*(-3.451329681590587444e-41 + T_*(-1.629936596105260491e-43)))))))))))))))))))) + Tref*(6.211050901462248372e-09 + T_*(3.874480667788168084e-10 + T_*(4.106851040300304659e-12 + T_*(-3.105169520741854561e-13 + T_*(6.522151398403065119e-16 + T_*(3.096283820459271858e-16 + T_*(-6.565926448866785715e-18 + T_*(-7.159450456455795076e-20 + T_*(4.894802418930987747e-21 + T_*(-6.001966927419548388e-23 + T_*(-1.513821153252045092e-24 + T_*(5.010070342130555091e-26 + T_*(3.162510217391503999e-28 + T_*(-1.770123180458273006e-29 + T_*(-1.053932297446173471e-31 + T_*(3.190827677655366054e-33 + T_*(3.180222107117949408e-35 + T_*(-1.648380290063498673e-37 + T_*(-3.706141650512771631e-39 + T_*(-1.917722264772635693e-41 + T_*(-3.362382426227650908e-44)))))))))))))))))))) + Tref*(1.263124849842055366e-10 + T_*(6.187412516523896449e-12 + T_*(4.856354852070540130e-14 + T_*(-9.099759444145425678e-15 + T_*(1.560190728375411453e-16 + T_*(7.844891934613453307e-18 + T_*(-3.045057457877816053e-19 + T_*(-2.301735187073350059e-22 + T_*(1.799768784739026120e-22 + T_*(-2.234656647103105002e-24 + T_*(-5.049062278648236422e-26 + T_*(1.234519814239864004e-27 + T_*(8.987122286706115913e-30 + T_*(-3.505893794831842691e-31 + T_*(-1.936267714478340583e-33 + T_*(5.615005967424700712e-35 + T_*(4.748626410980998077e-37 + T_*(-3.117187102765338827e-39 + T_*(-5.342048836982600040e-41 + T_*(-2.315823054010398044e-43 + T_*(-3.233422625384509680e-46)))))))))))))))))))) + Tref*(3.041933860088385168e-12 + T_*(7.093111999258697384e-14 + T_*(-1.844005764063040847e-15 + T_*(-3.012726184142518609e-18 + T_*(2.286804695745121757e-18 + T_*(-3.778946134964007883e-20 + T_*(-5.425688203547919150e-22 + T_*(1.720852091689116264e-23 + T_*(-2.132216029860537651e-25 + T_*(1.204097053514651607e-26 + T_*(9.674019166203633406e-29 + T_*(-1.310039385511127070e-29 + T_*(-3.441825993234164085e-32 + T_*(5.218559560287548561e-33 + T_*(2.792178449174765620e-35 + T_*(-9.909642936552052681e-37 + T_*(-1.027812531335395456e-38 + T_*(4.888180488713776030e-41 + T_*(1.228573493776009552e-42 + T_*(6.780669148124561191e-45 + T_*(1.273888703944060394e-47)))))))))))))))))))) + Tref*(4.269738908941117406e-14 + T_*(3.515691990558187008e-16 + T_*(-5.752406629775988591e-17 + T_*(2.935526497578685499e-18 + T_*(-1.396603096459919851e-20 + T_*(-3.076116403302916373e-21 + T_*(9.652802795654607107e-23 + T_*(9.479133640033158372e-26 + T_*(-6.298473009854447451e-26 + T_*(1.067398063166126933e-27 + T_*(1.747803610841354067e-29 + T_*(-6.553290148516108139e-31 + T_*(-3.304812034827328800e-33 + T_*(2.060835499621757621e-34 + T_*(1.070192724105747808e-36 + T_*(-3.514326016203337634e-38 + T_*(-3.229557610146469936e-40 + T_*(1.843938327239782823e-42 + T_*(3.752745971817626646e-44 + T_*(1.855294075718714751e-46 + T_*(3.111545317890359977e-49)))))))))))))))))))) + Tref*(3.187227602061177940e-16 + T_*(-1.653226486329436251e-18 + T_*(-5.503279650803687541e-19 + T_*(3.910889420859907195e-20 + T_*(-4.933223004188857164e-22 + T_*(-3.385127545168397706e-23 + T_*(1.323951437519895638e-24 + T_*(-2.175849457167253713e-27 + T_*(-7.767216266915100570e-28 + T_*(1.278597695041199424e-29 + T_*(2.085697980946302978e-31 + T_*(-6.977592433892922353e-33 + T_*(-3.774593366845591632e-35 + T_*(2.047178660783037856e-36 + T_*(1.050350400037973286e-38 + T_*(-3.355921373609339394e-40 + T_*(-2.957951585520604123e-42 + T_*(1.782466345297969279e-44 + T_*(3.384503662089840092e-46 + T_*(1.597754239120040422e-48 + T_*(2.533589690591361309e-51)))))))))))))))))))) + Tref*(9.638094015677524082e-19 + T_*(-1.825497478946932691e-20 + T_*(-1.687807799941253959e-21 + T_*(1.540362530250114278e-22 + T_*(-2.627978436697326027e-24 + T_*(-1.158894467996326174e-25 + T_*(5.277085797941572453e-27 + T_*(-1.725975868117368293e-29 + T_*(-2.913361578529600527e-30 + T_*(4.827457622910831529e-32 + T_*(7.681289388780240843e-34 + T_*(-2.437302887117087984e-35 + T_*(-1.370426782411350938e-37 + T_*(6.806940090614902811e-39 + T_*(3.495435520132081072e-41 + T_*(-1.081253869553308734e-42 + T_*(-9.298263251666167061e-45 + T_*(5.750522446000603233e-47 + T_*(1.045544411570244632e-48 + T_*(4.767127037305669495e-51 + T_*(7.211593011699459480e-54))))))))))))))))))))))))))))))  # noqa: E501
    else:
        thw = np.nan
    
    # Return theta-w converted to K
    return thw + 273.15

@vectorize(['float32(float32, float32)', 'float64(float64, float64)'], nopython=True)
def temp(p, thw):
    """
    Computes the temperature T at pressure p on a pseudoadiabat with
    wet-bulb potential temperature thw.

    Uses polynomial approximations from Moisseeva and Stull (2017) with
    revised coefficients.

    Moisseeva, N. and Stull, R., 2017. A noniterative approach to
        modelling moist thermodynamics. Atmospheric Chemistry and
        Physics, 17, 15037-15043.

    Args:
        p: pressure (Pa)
        thw: wet-bulb potential temperature (K)

    Returns:
        T: temperature (K)

    """

    # Convert p to hPa and theta-w to degC
    p_ = p / 100.
    thw_ = thw - 273.15

    # Compute T using Eq. 1-3 from Moisseeva & Stull 2017
    if (thw_ >= -70.) & (thw_ <= 50.) & (p_ <= 1100.) & (p_ >= 50.):
        thref = -1.958881611671661176e+02 + p_*(2.134884082821395079e+00 + p_*(-2.651475098307509368e-02 + p_*(2.861864119262733791e-04 + p_*(-2.298718394514268143e-06 + p_*(1.360057184923542422e-08 + p_*(-5.958196441636455271e-11 + p_*(1.938675375399162892e-13 + p_*(-4.665355621127693766e-16 + p_*(8.139597343471045903e-19 + p_*(-9.718027816571788133e-22 + p_*(6.514043622263483823e-25 + p_*(4.795894401108516600e-29 + p_*(-5.561331861642867047e-31 + p_*(4.256943236052971359e-34 + p_*(1.115187417957733097e-37 + p_*(-4.675607928134105329e-40 + p_*(4.189061674074321886e-43 + p_*(-1.989920659873727387e-46 + p_*(5.148437033814795851e-50 + p_*(-5.751272231517078191e-54))))))))))))))))))))  # noqa: E501
        T = -2.899089457107268331e+01 + thw_*(1.337227498242554491e+00 + thw_*(9.989220649709655911e-03 + thw_*(-5.289649585393284086e-05 + thw_*(-8.125516739581656903e-06 + thw_*(-1.669385809901756079e-07 + thw_*(3.902176729685648592e-09 + thw_*(2.785299448741561866e-10 + thw_*(1.199597501486574654e-12 + thw_*(-2.356495994204141054e-13 + thw_*(-3.754462622941184458e-15 + thw_*(1.109955443870932428e-16 + thw_*(2.958323602057082693e-18 + thw_*(-2.247001341245910925e-20 + thw_*(-1.185942142170470679e-21 + thw_*(-2.645697164120065566e-24 + thw_*(2.354624142321289223e-25 + thw_*(2.070711502559931296e-27 + thw_*(-1.458747561161565743e-29 + thw_*(-2.729648310305078289e-31 + thw_*(-1.030941535866486469e-33)))))))))))))))))))) + thref*(1.429869503550506904e+00 + thw_*(-7.879837833208863662e-03 + thw_*(-6.838366952421416926e-04 + thw_*(-1.598425851503747948e-05 + thw_*(2.249449259819928238e-07 + thw_*(3.397632056104877195e-08 + thw_*(7.042819999431954275e-10 + thw_*(-3.396305216284396052e-11 + thw_*(-1.427634441882554734e-12 + thw_*(1.718717725756761431e-14 + thw_*(1.351301204465966675e-15 + thw_*(-6.192018154861673091e-19 + thw_*(-7.436786948388566283e-19 + thw_*(-4.586031806307956854e-21 + thw_*(2.361621265751940082e-22 + thw_*(2.687010240026768440e-24 + thw_*(-3.700784758878172927e-26 + thw_*(-6.641106252235517576e-28 + thw_*(9.656001298499274765e-31 + thw_*(6.328645165577936637e-32 + thw_*(2.937789149798092732e-34)))))))))))))))))))) + thref*(5.040685977297330346e-03 + thw_*(-5.183478788794109284e-04 + thw_*(-8.614096880135471002e-06 + thw_*(6.838202302696602762e-07 + thw_*(4.744824589422048218e-08 + thw_*(4.117928705641196483e-10 + thw_*(-8.162969260401781373e-11 + thw_*(-2.506138383042399551e-12 + thw_*(6.408005845436154616e-14 + thw_*(3.422786758033482118e-15 + thw_*(-1.423870299706560812e-17 + thw_*(-2.347455893945982803e-18 + thw_*(-1.318085039469029337e-20 + thw_*(8.499042116124600158e-22 + thw_*(1.076807733584116111e-23 + thw_*(-1.338371226030613206e-25 + thw_*(-3.037207845698305137e-27 + thw_*(-2.280138198346503550e-30 + thw_*(2.951988306149798322e-31 + thw_*(2.237589772936888619e-33 + thw_*(4.618048611010569571e-36)))))))))))))))))))) + thref*(-1.693011573882441043e-04 + thw_*(-4.055505851887614839e-06 + thw_*(7.152523677763036439e-07 + thw_*(3.843257858703898518e-08 + thw_*(-5.844911310861310370e-10 + thw_*(-1.223121149866247755e-10 + thw_*(-1.922842158181238273e-12 + thw_*(1.860182907731545371e-13 + thw_*(5.585246327323852151e-15 + thw_*(-1.464431913286146844e-16 + thw_*(-6.559251795406730752e-18 + thw_*(4.899420092423094807e-20 + thw_*(4.199638168089739444e-21 + thw_*(7.896679384714861248e-24 + thw_*(-1.500870337486252050e-24 + thw_*(-1.219254495390460205e-26 + thw_*(2.649010861349269856e-28 + thw_*(3.763581851733364165e-30 + thw_*(-1.106962190807747265e-32 + thw_*(-4.004411643912248489e-34 + thw_*(-1.755651685348803267e-36)))))))))))))))))))) + thref*(-3.362873566158531171e-06 + thw_*(2.400942971282232631e-07 + thw_*(2.216103471465453210e-08 + thw_*(-2.358530382561856690e-10 + thw_*(-9.126436748430261388e-11 + thw_*(-1.931576827275146865e-12 + thw_*(1.740569037403095845e-13 + thw_*(6.367717531052867911e-15 + thw_*(-1.593228821759510933e-16 + thw_*(-8.645773823963254084e-18 + thw_*(5.183005959622386943e-20 + thw_*(6.216660172191152019e-21 + thw_*(2.280562673183416492e-23 + thw_*(-2.407055696000981067e-24 + thw_*(-2.497159694469028016e-26 + thw_*(4.310619140953843283e-28 + thw_*(7.767424624269287546e-30 + thw_*(-8.253668170995669652e-33 + thw_*(-8.265671014355856887e-34 + thw_*(-5.055332555603654578e-36 + thw_*(-6.866406703219498400e-39)))))))))))))))))))) + thref*(3.313006618121957636e-08 + thw_*(6.298294495254139702e-09 + thw_*(-1.508115170726802090e-10 + thw_*(-3.666058876697772524e-11 + thw_*(-6.929713447725405319e-13 + thw_*(1.001411867284538917e-13 + thw_*(3.999009200007800670e-15 + thw_*(-1.355140171228429127e-16 + thw_*(-7.891853868577618701e-18 + thw_*(8.201739075969596332e-20 + thw_*(8.157502196248929472e-21 + thw_*(1.872664694326145830e-24 + thw_*(-4.816139246993863831e-24 + thw_*(-3.251690295448622985e-26 + thw_*(1.596069969864303561e-27 + thw_*(1.911924811476329357e-29 + thw_*(-2.539650134731972660e-31 + thw_*(-4.766708118214056693e-33 + thw_*(5.905961735438100206e-36 + thw_*(4.542824189463173739e-37 + thw_*(2.146594442973875009e-39)))))))))))))))))))) + thref*(1.827646537676912968e-09 + thw_*(2.700624495348298076e-11 + thw_*(-1.693998063665866103e-11 + thw_*(-6.246467298925717754e-13 + thw_*(5.645404365776630832e-14 + thw_*(3.192962277411950210e-15 + thw_*(-8.101287993813953261e-17 + thw_*(-7.067390583742533053e-18 + thw_*(2.929925634625895208e-20 + thw_*(8.199180601829400231e-21 + thw_*(5.440328877403649776e-23 + thw_*(-5.211512128625378953e-24 + thw_*(-7.469192185405930370e-26 + thw_*(1.680194632500845717e-27 + thw_*(3.931613027193546259e-29 + thw_*(-1.570489589795977390e-31 + thw_*(-9.529584442335513237e-33 + thw_*(-4.564784845461340239e-35 + thw_*(7.845559059350718099e-37 + thw_*(9.429015911623818610e-39 + thw_*(2.969487264396173626e-41)))))))))))))))))))) + thref*(2.692489627401177037e-11 + thw_*(-8.866612549898191130e-13 + thw_*(-3.282133330510417204e-13 + thw_*(-2.176991467782678476e-15 + thw_*(1.525548282484946781e-15 + thw_*(3.255140898734812961e-17 + thw_*(-3.448238972506286259e-18 + thw_*(-9.819269470584681398e-20 + thw_*(4.103973882000636586e-21 + thw_*(1.421360768154288777e-22 + thw_*(-2.535930235573404443e-24 + thw_*(-1.138555920321440711e-25 + thw_*(6.357268121704545670e-28 + thw_*(5.168500299704030730e-29 + thw_*(8.966189300449154540e-32 + thw_*(-1.256207628001013337e-32 + thw_*(-8.343615181202799912e-35 + thw_*(1.324806169302695400e-36 + thw_*(1.409632581667273272e-38 + thw_*(-1.806982775263108798e-41 + thw_*(-4.102485528397047362e-43)))))))))))))))))))) + thref*(2.012525693579932692e-13 + thw_*(-1.409558681656955683e-14 + thw_*(-3.000478150434391517e-15 + thw_*(3.523413088054992200e-17 + thw_*(1.627592243174956061e-17 + thw_*(1.005263036983865846e-19 + thw_*(-4.256579318654112554e-20 + thw_*(-5.804877230475824782e-22 + thw_*(5.995951096343076474e-23 + thw_*(1.084604700494159426e-24 + thw_*(-4.758795393781846959e-26 + thw_*(-1.049404375984382912e-27 + thw_*(2.091294597612180069e-29 + thw_*(5.744517949902291411e-31 + thw_*(-4.386198422439627512e-33 + thw_*(-1.770452206683057335e-34 + thw_*(9.429173370089740969e-38 + thw_*(2.809453846841474016e-38 + thw_*(1.073364281318929982e-40 + thw_*(-1.723453344911116331e-42 + thw_*(-1.112769846261706298e-44)))))))))))))))))))) + thref*(7.785302726588919706e-16 + thw_*(-8.115422322369204709e-17 + thw_*(-1.359423616814330821e-17 + thw_*(3.494943093543307934e-19 + thw_*(8.118345806601954328e-20 + thw_*(-3.050691842782283411e-22 + thw_*(-2.304670197233203703e-22 + thw_*(-1.264239996599368699e-24 + thw_*(3.525780381700034179e-25 + thw_*(3.805080939256278211e-27 + thw_*(-3.081460621592729418e-28 + thw_*(-4.551988505380969105e-30 + thw_*(1.546975182182166271e-31 + thw_*(2.902341615631649949e-33 + thw_*(-4.149081096195920384e-35 + thw_*(-1.029080159047627957e-36 + thw_*(4.190976007612783375e-39 + thw_*(1.906088654394316995e-40 + thw_*(3.603634349189407222e-43 + thw_*(-1.430304392526457524e-44 + thw_*(-8.040338359871879916e-47)))))))))))))))))))) + thref*(1.239375080712697757e-18 + thw_*(-1.697253721082437881e-19 + thw_*(-2.457949980434437093e-20 + thw_*(9.128133448732742170e-22 + thw_*(1.568961363107645293e-22 + thw_*(-1.757664150666109165e-24 + thw_*(-4.703796383698215715e-25 + thw_*(2.947044230184508861e-29 + thw_*(7.581119354391254214e-28 + thw_*(4.806360277319273094e-30 + thw_*(-7.004664901833034528e-31 + thw_*(-7.627496323163632015e-33 + thw_*(3.757741880027517288e-34 + thw_*(5.601780195611557017e-36 + thw_*(-1.108150827912297810e-37 + thw_*(-2.199929486827426309e-39 + thw_*(1.405470896802169264e-41 + thw_*(4.459709728573167858e-43 + thw_*(3.991616449307566712e-46 + thw_*(-3.666103046079535075e-47 + thw_*(-1.943440793139680257e-49))))))))))))))))))))))))))))))  # noqa: E501
    else:
        T = np.nan

    # Return T converted to K
    return T + 273.15

# ── Helper functions ─────────────────────────────────────────────────────────

def effective_gas_constant(q, qt=None):
    """
    Computes effective gas constant for moist air.

    Args:
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)

    Returns:
        Rm (float or ndarray): effective gas constant (J/kg/K)

    """
    if qt is None:
        # (Eq. 12 from Warren 2025)
        Rm = (1 - q) * Rd + q * Rv  # = Rd * (1 - q + q / eps)
    else:
        # (Eq. 10 from Warren 2025)
        Rm = (1 - qt) * Rd + q * Rv  # = Rd * (1 - qt + q / eps)

    return Rm

def effective_specific_heat(q, qt=None, omega=0.0):
    """
    Computes effective isobaric specific heat for moist air.

    Args:
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)
        omega (float or ndarray, optional): ice fraction

    Returns:
        cpm (float or ndarray): effective isobaric specific heat (J/kg/K)

    """
    if qt is None:
        # (Eq. 17 from Warren 2025)
        cpm = (1 - q) * cpd + q * cpv  # = cpd * (1 - q + q / lambda),
                                       # where lambda = cpd / cpv
    else:
        # (Eq. 16 from Warren 2025)
        ql = (1 - omega) * (qt - q)  # liquid water mass fraction
        qi = omega * (qt - q)        # ice water mass fraction
        cpm = (1 - qt) * cpd + q * cpv + ql * cpl + qi * cpi

    return cpm


def latent_heat_of_vaporisation(T):
    """
    Computes latent heat of vaporisation for a given temperature.

    Args:
        T (float or ndarray): temperature (K)

    Returns:
        Lv (float or ndarray): latent heat of vaporisation (J/kg)

    """
    # (Eq. 23 from Warren 2025)
    Lv = Lv0 + (cpv - cpl) * (T - T0)

    return Lv


def latent_heat_of_sublimation(T):
    """
    Computes latent heat of sublimation for a given temperature.

    Args:
        T (float or ndarray): temperature (K)

    Returns:
        Ls (float or ndarray): latent heat of sublimation (J/kg)

    """
    # (Eq. 24 from Warren 2025)
    Ls = Ls0 + (cpv - cpi) * (T - T0)

    return Ls


def mixed_phase_latent_heat(T, omega):
    """
    Computes mixed-phase latent heat for a given temperature and ice fraction
    using equations from Warren (2025).

    Args:
        T (float or ndarray): temperature (K)
        omega (float or ndarray): ice fraction

    Returns:
        Lx (float or ndarray): mixed-phase latent heat (J/kg)

    """

    # Compute mixed-phase specific heat
    # (Eq. 30 from Warren 2025)
    cpx = (1 - omega) * cpl + omega * cpi

    # Compute mixed-phase latent heat at the triple point
    # (Eq. 31 from Warren 2025)
    Lx0 = (1 - omega) * Lv0 + omega * Ls0  # = Lv0 + omega * Lf0

    # Compute mixed-phase latent heat
    # (Eq. 32 from Warren 2025)
    Lx = Lx0 + (cpv - cpx) * (T - T0)  # = (1 - omega) * Lv + omega * Ls
                                       # = Lv + omega * Lf

    return Lx


def vapour_pressure(p, q, qt=None):
    """
    Computes vapour pressure from pressure and specific humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)

    Returns:
        e (float or ndarray): vapour pressure (Pa)

    """
    # (Eq. 15 from Warren 2025)
    if qt is None:
        e = p * q / (eps * (1 - q) + q)
    else:
        e = p * q / (eps * (1 - qt) + q)

    return e


def saturation_vapour_pressure(T, phase='liquid', omega=0.0):
    """
    Computes saturation vapour pressure (SVP) for a given temperature using
    equations from Ambaum (2020) and Warren (2025).

    Args:
        T (float or ndarray): temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        es (float or ndarray): saturation vapour pressure (Pa)

    """
   
    if phase == 'liquid':
        
        # Compute latent heat of vaporisation
        Lv = latent_heat_of_vaporisation(T)

        # Compute SVP over liquid water
        # (Eq. 26 from Warren 2025; cf. Eq. 13 from Ambaum 2020)
        es = es0 * np.power((T0 / T), ((cpl - cpv) / Rv)) * \
            np.exp((Lv0 / (Rv * T0)) - (Lv / (Rv * T)))
        
    elif phase == 'ice':
        
        # Compute latent heat of sublimation
        Ls = latent_heat_of_sublimation(T)

        # Compute SVP over ice
        # (Eq. 27 from Warren 2025; cf. Eq. 17 from Ambaum 2020)
        es = es0 * np.power((T0 / T), ((cpi - cpv) / Rv)) * \
            np.exp((Ls0 / (Rv * T0)) - (Ls / (Rv * T)))
        
    elif phase == 'mixed':
        
        # Compute mixed-phase specific heat
        # (Eq. 30 from Warren 2025)
        cpx = (1 - omega) * cpl + omega * cpi
        
        # Compute mixed-phase latent heat at the triple point
        # (Eq. 31 from Warren 2025)
        Lx0 = (1 - omega) * Lv0 + omega * Ls0

        # Compute mixed-phase latent heat
        # (Eq. 32 from Warren 2025)
        Lx = Lx0 + (cpv - cpx) * (T - T0)
        
        # Compute mixed-phase SVP
        # (Eq. 29 from Warren 2025)
        es = es0 * np.power((T0 / T), ((cpx - cpv) / Rv)) * \
            np.exp((Lx0 / (Rv * T0)) - (Lx / (Rv * T)))
        
    else:

        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    return es


def saturation_specific_humidity(p, T, qt=None, phase='liquid', omega=0.0):
    """
    Computes saturation specific humidity from pressure and temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        qs (float or ndarray): saturation specific humidity (kg/kg)

    """
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    if qt is None:
        # (Eq. 14 from Warren 2025, with qv = qt = qs and e = es)
        qs = eps * es / (p - (1 - eps) * es)
    else:
        # (Eq. 14 from Warren 2025, with qv = qs and e = es)
        qs  = (1 - qt) * eps * es / (p - es)

    return qs


def relative_humidity(p, T, q, qt=None, phase='liquid', omega=0.0):
    """
    Computes relative humidity with respect to specified phase from pressure, 
    temperature, and specific humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation 
            (default is 0.0)
        
    Returns:
        RH (float or ndarray): relative humidity (fraction)

    """
    e = vapour_pressure(p, q, qt=qt)
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    RH = e / es

    return RH


def ice_fraction(Tstar, phase='mixed'):
    """
    Computes ice fraction given temperature at saturation using nonlinear
    parameterisation of Warren (2025).

    Args:
        Tstar (float or ndarray): temperature at saturation (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'mixed')

    Returns:
        omega (float or ndarray): ice fraction

    """

    # Ensure that Tstar is an array
    Tstar = np.atleast_1d(Tstar)

    # Compute the ice fraction
    # (Eq. 6 from Warren 2025, with T = Tstar)
    if phase == 'liquid':
        omega = np.zeros_like(Tstar)  # ice fraction is zero
    elif phase == 'ice':
        omega = np.ones_like(Tstar)  # ice fraction is one
    elif phase == 'mixed':
        omega = 0.5 * (1 - np.cos(np.pi * ((T_liq - Tstar) / (T_liq - T_ice))))
        omega[Tstar <= T_ice] = 1.0
        omega[Tstar >= T_liq] = 0.0
    else:
        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    if omega.size == 1:
        omega = omega.item()

    return omega

def ice_fraction_derivative(Tstar, phase='mixed'):
    """
    Computes derivative of ice fraction with respect to temperature at
    saturation using nonlinear parameterisation of Warren (2025).
    
    Args:
        Tstar (float or ndarray): temperature at saturation (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'mixed')

    Returns:
        domega_dTstar (float or ndarray): derivative of ice fraction (K^-1)
       
    """

    # Ensure that Tstar is an array
    Tstar = np.atleast_1d(Tstar)

    # Compute the derivative of the ice fraction
    # (derivative of Eq. 6 from Warren 2025, with T = Tstar)
    if phase == 'liquid' or phase == 'ice':
        domega_dTstar = np.zeros_like(Tstar)  # derivative is zero
    elif phase == 'mixed':
        domega_dTstar = -0.5 * (np.pi / (T_liq - T_ice)) * \
                np.sin(np.pi * ((T_liq - Tstar) / (T_liq - T_ice)))
        domega_dTstar[(Tstar <= T_ice) | (Tstar >= T_liq)] = 0.0
    else:
        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    if domega_dTstar.size == 1:
        domega_dTstar = domega_dTstar.item()

    return domega_dTstar


@vectorize(['float32(float32)', 'float64(float64)'], nopython=True)
def _lambertw(x):
    """
    Evaluates the lower branch of the Lambert-W function using PSEM
    approximation from Vazquez-Leal et al. (2019).

    Args:
        x (float): dependent variable

    Returns:
        y (float): Lambert W function (lower branch)

    """

    y = np.nan

    # W_{-1,1}
    # (Eq. 15 from Vazquez-Leal et al. 2019)
    if (x >= -0.3678794411714423) and (x < -0.34):
        a1 = -7.874564067684664
        a2 = -63.11879948166995
        a3 = -168.6110850408981
        a4 = -150.1089086912451
        b1 = 15.97679839497612
        b2 = 98.26612857148953
        b3 = 293.9558944644677
        b4 = 430.4471947824411
        b5 = 247.8576700279611
        alpha = x * (a1 + x * (a2 + x * (a3 + x * a4)))
        beta = 1. + x * (b1 + x * (b2 + x * (b3 + x * (b4 + x * b5))))
        y = (alpha / beta) * (x + np.exp(-1.)) - 1.

    # W_{-1,2}
    # (Eq. 16 from Vazquez-Leal et al. 2019)
    if (x >= -0.34) and (x < -0.1):
        a1 = -1362.78381643109
        a2 = -1386.04132570149
        a3 = 11892.1649836015
        a4 = 16904.0507511421
        b1 = 251.440197724561
        b2 = -1264.99554712435
        b3 = -5687.63429510978
        b4 = -2639.24130979048
        y = (x * (a1 + x * (a2 + x * (a3 + x * a4)))) / \
            (1. + x * (b1 + x * (b2 + x * (b3 + x * b4))))

    # W_{-1,3}
    # (Eq. 17-18 from Vazquez-Leal et al. 2019)
    if (x >= -0.1) and (x < 0.):
        a1 = 1.01999365162218
        a2 = -12.6917365519443
        a3 = -45.1506015092455
        b1 = -22.9809693297808
        b2 = -104.692066099727
        b3 = -95.2085341727207
        k0 = (x * (a1 + x * (a2 + x * a3))) / \
             (1. + x * (b1 + x * (b2 + x * b3)))
        k1 = np.log(-x)
        k2 = k1 - np.log(-k1) + np.log(-k1) / k1
        y = k0 + k2

    # Iterate once to improve accuracy
    # (Eq. 30-32 from Vazquez-Leal et al. 2019)
    z = np.log(x / y) - y
    t = 2. * (1. + y) * (1. + y + (2./3.) * z)
    e = (z / (1. + y)) * (t - z) / (t - 2. * z)
    y = y * (1. + e)

    return y


def lifting_condensation_level(p, T, q, phase='liquid', limit=True):
    """
    Computes pressure and temperature at the lifting condensation level (LCL)
    using equations from Romps (2017), as presented in Warren (2025). If
    phase='ice', the pressure and temperature at the lifting deposition level
    (LDL) are calculated. If phase='mixed', the pressure and temperature at
    the lifting saturation level (LSL) are calculated. For simplicity, the
    symbols T_lcl and p_lcl are used for all three of these levels.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        limit (bool, optional): flag indicating whether to limit the LCL
            temperature and pressure to ensure that they does not exceed the
            initial values (default is True)

    Returns:
        p_lcl (float or ndarray): pressure at the LCL (Pa)
        T_lcl (float or ndarray): temperature at the LCL (K)

    """

    # Compute effective gas constant and specific heat
    Rm = effective_gas_constant(q)
    cpm = effective_specific_heat(q)

    if phase == 'liquid':

        # Compute relative humidity with respect to liquid water
        RH = relative_humidity(p, T, q, phase='liquid')

        # Compute temperature at the LCL using Lambert W function
        # (Eq. 44 and 46-47 from Warren 2025; cf. Eq. 22a,d-f from Romps 2017)
        beta = cpm / Rm + (cpl - cpv) / Rv
        alpha = -(1 / beta) * (Lv0 + (cpl - cpv) * T0) / Rv
        fn = np.power(RH, (1 / beta)) * (alpha / T) * np.exp(alpha / T)
        #W = lambertw(fn, k=-1).real
        W = _lambertw(fn)
        T_lcl = alpha / W

    elif phase == 'ice':

        # Compute relative humidity with respect to ice
        RH = relative_humidity(p, T, q, phase='ice')

        # Compute temperature at the LDL using Lambert W function
        # (Eq. 48 and 50-51 from Warren 2025; cf. Eq. 23a,d-f from Romps 2017)
        beta = cpm / Rm + (cpi - cpv) / Rv
        alpha = -(1 / beta) * (Ls0 + (cpi - cpv) * T0) / Rv
        fn = np.power(RH, (1 / beta)) * (alpha / T) * np.exp(alpha / T)
        #W = lambertw(fn, k=-1).real
        W = _lambertw(fn)
        T_lcl = alpha / W

    elif phase == 'mixed':

        # Initialise the LSL temperature as the temperature
        T_lcl = T

        # Iterate to convergence
        converged = False
        count = 0
        while not converged:

            # Update the previous LSL temperature value
            T_lcl_prev = T_lcl

            # Compute the ice fraction
            omega = ice_fraction(T_lcl)

            # Compute mixed-phase relative humidity
            RH = relative_humidity(p, T, q, phase='mixed', omega=omega)

            # Compute mixed-phase specific heat
            # (Eq. 30 from Warren 2025)
            cpx = (1 - omega) * cpl + omega * cpi

            # Compute mixed-phase latent heat at the triple point
            # (Eq. 31 from Warren 2025)
            Lx0 = (1 - omega) * Lv0 + omega * Ls0

            # Compute temperature at the LSL using Lambert-W function
            # (Eq. 52 and 54-55 from Warren 2025)
            beta = cpm / Rm + (cpx - cpv) / Rv
            alpha = -(1 / beta) * (Lx0 + (cpx - cpv) * T0) / Rv
            fn = np.power(RH, (1 / beta)) * (alpha / T) * np.exp(alpha / T)
            #W = lambertw(fn, k=-1).real  # -1 branch because cpx > cpv
            W = _lambertw(fn)
            T_lcl = alpha / W

            # Check if solution has converged
            if np.max(np.abs(T_lcl - T_lcl_prev)) < precision:
                converged  = True
            else:
                count += 1
                if count == max_n_iter:
                    print(f"LSL temperature not converged after {max_n_iter} iterations")
                    break

    else:

        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    # Compute pressure at the LCL / LDL / LSL
    # (Eq. 45 from Warren 2025)
    p_lcl = p * np.power((T_lcl / T), (cpm / Rm))

    if limit:
        T_lcl = np.minimum(T_lcl, T)
        p_lcl = np.minimum(p_lcl, p)

    return p_lcl, T_lcl


def dry_adiabatic_lapse_rate(p, T, q):
    """
    Computes dry adiabatic lapse rate in pressure coordinates.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        dT_dp (float or ndarray): dry adiabatic lapse rate (K/Pa)

    """

    # Compute the effective gas constant
    Rm = effective_gas_constant(q)

    # Compute the effective specific heat
    cpm = effective_specific_heat(q)

    # Compute dry adiabatic lapse rate
    dT_dp = (1 / p) * (Rm * T / cpm)

    return dT_dp


def pseudoadiabatic_lapse_rate(p, T, phase='liquid'):
    """
    Computes pseudoadiabatic lapse rate in pressure coordinates using equations
    from Warren (2025).

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')

    Returns:
        dT_dp (float or ndarray): pseudoadiabatic lapse rate (K/Pa)

    """

    # Set the ice fraction
    omega = ice_fraction(T, phase=phase)

    # Compute saturation specific humidity
    qs = saturation_specific_humidity(p, T, phase=phase, omega=omega)

    # Compute Q term
    # (Eq. 71 from Warren 2025)
    Q = qs * (1 - qs + qs / eps)

    # Compute the effective gas constant
    Rm = effective_gas_constant(qs)

    # Compute the effective specific heat
    cpm = effective_specific_heat(qs)

    if phase == 'liquid':

        # Compute latent heat of vaporisation
        Lv = latent_heat_of_vaporisation(T)

        # Compute liquid pseudoadiabatic lapse rate
        # (Eq. 74 from Warren 2025)
        dT_dp = (1 / p) * (Rm * T + Lv * Q) / \
            (cpm + (Lv**2 * Q) / (Rv * T**2))

    elif phase == 'ice':

        # Compute latent heat of sublimation
        Ls = latent_heat_of_sublimation(T)

        # Compute ice pseudoadiabatic lapse rate
        # (Eq. 72 from Warren 2025, with omega = 1, so that Lx = Ls and
        # domega_dT = 0)
        dT_dp = (1 / p) * (Rm * T + Ls * Q) / \
            (cpm + (Ls**2 * Q) / (Rv * T**2))

    elif phase == 'mixed':

        # Compute the derivative of omega with respect to temperature
        domega_dT = ice_fraction_derivative(T)

        # Compute mixed-phase latent heat
        Lx = mixed_phase_latent_heat(T, omega)

        # Compute saturation vapour pressues over liquid and ice
        esl = saturation_vapour_pressure(T, phase='liquid')
        esi = saturation_vapour_pressure(T, phase='ice')

        # Compute mixed-phase pseudoadiabatic lapse rate
        # (Eq. 72 from Warren 2025)
        dT_dp = (1 / p) * (Rm * T + Lx * Q) / \
            (cpm + (Lx**2 * Q) / (Rv * T**2) +
             Lx * Q * np.log(esi / esl) * domega_dT)

    return dT_dp


def follow_moist_adiabat(
    pi, pf, Ti, qt=None, phase='liquid',
    pseudo=True, polynomial=True, explicit=False, dp=500.0
    ):
    """
    Computes parcel temperature following a saturated adiabat or pseudoadiabat.
    For descending parcels, a pseudoadiabat is always used. By default,
    pseudoadiabatic calculations use polynomial fits for fast calculations, but
    can optionally use direct integration. Saturated adiabatic ascent can only
    be performed using direct integration (for now). At present, polynomial
    fits are only available for liquid pseudoadiabats.

    Args:
        pi (float or ndarray): initial pressure (Pa)
        pf (float or ndarray): final pressure (Pa)
        Ti (float or ndarray): initial temperature (K)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None; required for saturated adiabatic ascent)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        pseudo (bool): flag indicating whether to perform pseudoadiabatic
            parcel ascent (default is True)
        polynomial (bool, optional): flag indicating whether to use polynomial
            fits to pseudoadiabats (default is True)
        explicit (bool, optional): flag indicating whether to use explicit
            integration of lapse rate equation (default is False)
        dp (float, optional): pressure increment for integration of lapse rate
            equation (default is 500 Pa = 5 hPa)

    Returns:
        Tf (float or ndarray): final temperature (K)

    """

    if pseudo and polynomial:

        if phase != 'liquid':
            raise ValueError(
                """Polynomial fits have yet to be created for ice and 
                mixed-phase pseudoadiabats. Calculations must be performed 
                using direct integration by setting polynomial=False."""
                )

        # Compute the wet-bulb potential temperature of the pseudoadiabat
        # that passes through (pi, Ti)
        thw = wbpt(pi, Ti)

        # Compute the temperature on this pseudoadiabat at pf
        Tf = temp(pf, thw)

        # Ensure that final temperature is equal to initial temperature where
        # initial and final pressures are equal (this is not guaraneteed due to
        # small errors introduced in polynomial fits)
        Tf = np.where(pf == pi, Ti, Tf)

    else:

        if not pseudo and qt is None:
            raise ValueError('qt is required for saturated adiabatic ascent')

        pi = np.atleast_1d(pi)
        pf = np.atleast_1d(pf)
        Ti = np.atleast_1d(Ti)

        if Ti.size > 1:
            # multiple initial temperature values
            if pi.size == 1:
                # single initial pressure value
                pi = np.full_like(Ti, pi)
            if pf.size == 1:
                # single final pressure value
                pf = np.full_like(Ti, pf)

        # Set the pressure increment based on whether the parcel is ascending
        # or descending
        dp = np.abs(dp)  # make sure pressure increment is positive
        ascending = (pf < pi)
        descending = np.logical_not(ascending)
        dp = np.where(ascending, -dp, dp)

        # Initialise the pressure and temperature at level 2
        p2 = np.copy(pi)
        T2 = np.copy(Ti)

        # Create an array to store the lapse rate
        dT_dp = np.zeros_like(p2)

        #print(np.min(p2), np.max(p2))
        #print(np.count_nonzero(ascending), np.count_nonzero(descending))
        #print(np.min(dp), np.max(dp))

        # Loop over pressure increments
        while np.nanmax(np.abs(p2 - pf)) > 0.0:

            # Set level 1 values
            p1 = p2
            T1 = T2

            # Update the pressure at level 2
            p2 = p1 + dp

            # Make sure we haven't overshot final pressure level
            if np.any(ascending):
                p2[ascending] = np.maximum(p2[ascending], pf[ascending])
            if np.any(descending):
                p2[descending] = np.minimum(p2[descending], pf[descending])

            #print(np.min(p2), np.max(p2))

            # Compute the pressure at the layer mid-point (in log-space)
            pmid = np.sqrt(p1 * p2)  # = exp(0.5 * (log(p1) + log(p2)))

            # Get initial estimate for the temperature at level 2 by following
            # a dry adiabat (ignoring the contribution of moisture)
            dT_dp = dry_adiabatic_lapse_rate(p1, T1, 0.0)
            #T2 = T1 + dT_dp * (p2 - p1)
            T2 = T1 + pmid * dT_dp * np.log(p2 / p1)  # pmid * dT/dp = dT/dlnp

            # Iterate to get the new temperature at level 2
            converged = False
            count = 0
            while not converged:

                # Update the previous level 2 temperature
                T2_prev = T2

                # Compute the temperature at the layer mid-point
                Tmid = 0.5 * (T1 + T2)

                # Compute the lapse rate for ascending parcels
                if np.any(ascending):
                    if pseudo:
                        dT_dp[ascending] = pseudoadiabatic_lapse_rate(
                            pmid[ascending], Tmid[ascending],
                            phase=phase
                            )
                    else:
                        dT_dp[ascending] = saturated_adiabatic_lapse_rate(
                            pmid[ascending], Tmid[ascending], qt[ascending],
                            phase=phase
                            )

                # Compute the lapse rate for descending parcels
                if np.any(descending):
                    dT_dp[descending] = pseudoadiabatic_lapse_rate(
                        pmid[descending], Tmid[descending],
                        phase=phase
                        )

                # Update the level 2 temperature
                #T2 = T1 + dT_dp * (p2 - p1)
                T2 = T1 + pmid * dT_dp * np.log(p2 / p1)  # pmid * dT/dp = dT/dlnp

                # Check if the solution has converged
                if np.nanmax(np.abs(T2 - T2_prev)) < precision:
                    converged = True
                else:
                    count += 1
                    if count == max_n_iter:
                        # should converge in just a couple of iterations
                        # provided pinc is not too large
                        print(f'Not converged after {max_n_iter} iterations')
                        break

                if explicit:
                    # skip additional iterations
                    converged = True

            #print(np.min(p1), np.min(p2), count)

        # Set the final temperature
        Tf = T2

    return Tf


# ── Target function ───────────────────────────────────────────────────────────

def pseudo_wet_bulb_temperature(
    p, T, q, phase='liquid', method='NEWT', limit=True
    ):
    """
    Computes pseudo (a.k.a. adiabatic) wet-bulb temperature using method
    outlined in section 7 of Warren (2025).

    Pseudo wet-bulb temperature is the temperature of a parcel of air lifted
    adiabatically to saturation and then brought pseudoadiabatically at
    saturation back to its original pressure.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        method (str, optional): method used to perform calculation (valid
            options are 'iterative' or 'NEWT'; default is 'NEWT')
        limit (bool, optional): flag indicating whether to limit the wet-bulb
            temperature to ensure that it does not exceed the temperature
            (default is True)

    Returns:
        Tw (float or ndarray): pseudo wet-bulb temperature (K)

    """

    if method == 'iterative':
        polynomial = False
    elif method == 'NEWT':
        polynomial = True
    else:
        raise ValueError("method must be either 'iterative' or 'NEWT'")

    # Get pressure and temperature at the LCL
    p_lcl, T_lcl = lifting_condensation_level(p, T, q, phase=phase)

    # Follow a pseudoadiabat from the LCL to the original pressure
    Tw = follow_moist_adiabat(p_lcl, p, T_lcl, phase=phase, pseudo=True,
                              polynomial=polynomial)

    if limit:
        Tw = np.minimum(Tw, T)

    if not np.isscalar(Tw) and Tw.size == 1:
        Tw = Tw.item()

    return Tw