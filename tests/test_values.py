import numpy as np

x = np.linspace(-0.7, 0, 200)

EXPECTED_Y_ARRAY = (
    (1e-6 * np.exp(-((x + 0.3) ** 2 / (2 * 0.05 ** 2))))
    + (6e-6 * np.exp(-((x + 0.7) ** 2 / (2 * 0.01 ** 2))))
    + -1e-6 * x
    + 1e-6
)

EXPECTED_Y_FITTING_MATH = np.array(
    [
        7.70000000e-06,
        7.33652911e-06,
        6.37761372e-06,
        5.12767223e-06,
        3.91567074e-06,
        2.96013732e-06,
        2.32586340e-06,
        1.96483935e-06,
        1.78629565e-06,
        1.70831764e-06,
        1.67716358e-06,
        1.66467209e-06,
        1.65860006e-06,
        1.65444409e-06,
        1.65078627e-06,
        1.64724159e-06,
        1.64371939e-06,
        1.64020111e-06,
        1.63668343e-06,
        1.63316583e-06,
        1.62964824e-06,
        1.62613065e-06,
        1.62261307e-06,
        1.61909548e-06,
        1.61557789e-06,
        1.61206030e-06,
        1.60854272e-06,
        1.60502513e-06,
        1.60150755e-06,
        1.59798997e-06,
        1.59447239e-06,
        1.59095482e-06,
        1.58743725e-06,
        1.58391970e-06,
        1.58040216e-06,
        1.57688464e-06,
        1.57336716e-06,
        1.56984972e-06,
        1.56633235e-06,
        1.56281507e-06,
        1.55929793e-06,
        1.55578097e-06,
        1.55226428e-06,
        1.54874794e-06,
        1.54523211e-06,
        1.54171697e-06,
        1.53820276e-06,
        1.53468983e-06,
        1.53117863e-06,
        1.52766974e-06,
        1.52416396e-06,
        1.52066230e-06,
        1.51716609e-06,
        1.51367705e-06,
        1.51019738e-06,
        1.50672989e-06,
        1.50327816e-06,
        1.49984669e-06,
        1.49644112e-06,
        1.49306848e-06,
        1.48973746e-06,
        1.48645874e-06,
        1.48324536e-06,
        1.48011315e-06,
        1.47708118e-06,
        1.47417224e-06,
        1.47141345e-06,
        1.46883675e-06,
        1.46647958e-06,
        1.46438541e-06,
        1.46260440e-06,
        1.46119390e-06,
        1.46021898e-06,
        1.45975285e-06,
        1.45987707e-06,
        1.46068175e-06,
        1.46226538e-06,
        1.46473454e-06,
        1.46820323e-06,
        1.47279190e-06,
        1.47862612e-06,
        1.48583488e-06,
        1.49454838e-06,
        1.50489552e-06,
        1.51700087e-06,
        1.53098134e-06,
        1.54694242e-06,
        1.56497416e-06,
        1.58514701e-06,
        1.60750747e-06,
        1.63207379e-06,
        1.65883186e-06,
        1.68773140e-06,
        1.71868259e-06,
        1.75155337e-06,
        1.78616750e-06,
        1.82230362e-06,
        1.85969528e-06,
        1.89803224e-06,
        1.93696293e-06,
        1.97609824e-06,
        2.01501658e-06,
        2.05327010e-06,
        2.09039206e-06,
        2.12590513e-06,
        2.15933050e-06,
        2.19019745e-06,
        2.21805327e-06,
        2.24247308e-06,
        2.26306938e-06,
        2.27950099e-06,
        2.29148107e-06,
        2.29878399e-06,
        2.30125077e-06,
        2.29879298e-06,
        2.29139494e-06,
        2.27911405e-06,
        2.26207944e-06,
        2.24048876e-06,
        2.21460337e-06,
        2.18474209e-06,
        2.15127361e-06,
        2.11460793e-06,
        2.07518703e-06,
        2.03347513e-06,
        1.98994877e-06,
        1.94508706e-06,
        1.89936235e-06,
        1.85323154e-06,
        1.80712831e-06,
        1.76145638e-06,
        1.71658395e-06,
        1.67283934e-06,
        1.63050800e-06,
        1.58983070e-06,
        1.55100294e-06,
        1.51417554e-06,
        1.47945613e-06,
        1.44691155e-06,
        1.41657099e-06,
        1.38842956e-06,
        1.36245235e-06,
        1.33857865e-06,
        1.31672629e-06,
        1.29679588e-06,
        1.27867493e-06,
        1.26224172e-06,
        1.24736883e-06,
        1.23392627e-06,
        1.22178423e-06,
        1.21081546e-06,
        1.20089712e-06,
        1.19191230e-06,
        1.18375115e-06,
        1.17631165e-06,
        1.16950009e-06,
        1.16323123e-06,
        1.15742830e-06,
        1.15202281e-06,
        1.14695414e-06,
        1.14216912e-06,
        1.13762149e-06,
        1.13327128e-06,
        1.12908428e-06,
        1.12503138e-06,
        1.12108799e-06,
        1.11723352e-06,
        1.11345081e-06,
        1.10972568e-06,
        1.10604649e-06,
        1.10240372e-06,
        1.09878968e-06,
        1.09519816e-06,
        1.09162420e-06,
        1.08806386e-06,
        1.08451401e-06,
        1.08097221e-06,
        1.07743654e-06,
        1.07390553e-06,
        1.07037801e-06,
        1.06685313e-06,
        1.06333021e-06,
        1.05980873e-06,
        1.05628834e-06,
        1.05276872e-06,
        1.04924969e-06,
        1.04573107e-06,
        1.04221274e-06,
        1.03869464e-06,
        1.03517669e-06,
        1.03165885e-06,
        1.02814108e-06,
        1.02462337e-06,
        1.02110570e-06,
        1.01758806e-06,
        1.01407043e-06,
        1.01055282e-06,
        1.00703521e-06,
        1.00351761e-06,
        1.00000002e-06,
    ]
)

EXPECTED_X_FITTING_MATH = np.array(
    [
        -0.7,
        -0.69648241,
        -0.69296482,
        -0.68944724,
        -0.68592965,
        -0.68241206,
        -0.67889447,
        -0.67537688,
        -0.6718593,
        -0.66834171,
        -0.66482412,
        -0.66130653,
        -0.65778894,
        -0.65427136,
        -0.65075377,
        -0.64723618,
        -0.64371859,
        -0.64020101,
        -0.63668342,
        -0.63316583,
        -0.62964824,
        -0.62613065,
        -0.62261307,
        -0.61909548,
        -0.61557789,
        -0.6120603,
        -0.60854271,
        -0.60502513,
        -0.60150754,
        -0.59798995,
        -0.59447236,
        -0.59095477,
        -0.58743719,
        -0.5839196,
        -0.58040201,
        -0.57688442,
        -0.57336683,
        -0.56984925,
        -0.56633166,
        -0.56281407,
        -0.55929648,
        -0.55577889,
        -0.55226131,
        -0.54874372,
        -0.54522613,
        -0.54170854,
        -0.53819095,
        -0.53467337,
        -0.53115578,
        -0.52763819,
        -0.5241206,
        -0.52060302,
        -0.51708543,
        -0.51356784,
        -0.51005025,
        -0.50653266,
        -0.50301508,
        -0.49949749,
        -0.4959799,
        -0.49246231,
        -0.48894472,
        -0.48542714,
        -0.48190955,
        -0.47839196,
        -0.47487437,
        -0.47135678,
        -0.4678392,
        -0.46432161,
        -0.46080402,
        -0.45728643,
        -0.45376884,
        -0.45025126,
        -0.44673367,
        -0.44321608,
        -0.43969849,
        -0.4361809,
        -0.43266332,
        -0.42914573,
        -0.42562814,
        -0.42211055,
        -0.41859296,
        -0.41507538,
        -0.41155779,
        -0.4080402,
        -0.40452261,
        -0.40100503,
        -0.39748744,
        -0.39396985,
        -0.39045226,
        -0.38693467,
        -0.38341709,
        -0.3798995,
        -0.37638191,
        -0.37286432,
        -0.36934673,
        -0.36582915,
        -0.36231156,
        -0.35879397,
        -0.35527638,
        -0.35175879,
        -0.34824121,
        -0.34472362,
        -0.34120603,
        -0.33768844,
        -0.33417085,
        -0.33065327,
        -0.32713568,
        -0.32361809,
        -0.3201005,
        -0.31658291,
        -0.31306533,
        -0.30954774,
        -0.30603015,
        -0.30251256,
        -0.29899497,
        -0.29547739,
        -0.2919598,
        -0.28844221,
        -0.28492462,
        -0.28140704,
        -0.27788945,
        -0.27437186,
        -0.27085427,
        -0.26733668,
        -0.2638191,
        -0.26030151,
        -0.25678392,
        -0.25326633,
        -0.24974874,
        -0.24623116,
        -0.24271357,
        -0.23919598,
        -0.23567839,
        -0.2321608,
        -0.22864322,
        -0.22512563,
        -0.22160804,
        -0.21809045,
        -0.21457286,
        -0.21105528,
        -0.20753769,
        -0.2040201,
        -0.20050251,
        -0.19698492,
        -0.19346734,
        -0.18994975,
        -0.18643216,
        -0.18291457,
        -0.17939698,
        -0.1758794,
        -0.17236181,
        -0.16884422,
        -0.16532663,
        -0.16180905,
        -0.15829146,
        -0.15477387,
        -0.15125628,
        -0.14773869,
        -0.14422111,
        -0.14070352,
        -0.13718593,
        -0.13366834,
        -0.13015075,
        -0.12663317,
        -0.12311558,
        -0.11959799,
        -0.1160804,
        -0.11256281,
        -0.10904523,
        -0.10552764,
        -0.10201005,
        -0.09849246,
        -0.09497487,
        -0.09145729,
        -0.0879397,
        -0.08442211,
        -0.08090452,
        -0.07738693,
        -0.07386935,
        -0.07035176,
        -0.06683417,
        -0.06331658,
        -0.05979899,
        -0.05628141,
        -0.05276382,
        -0.04924623,
        -0.04572864,
        -0.04221106,
        -0.03869347,
        -0.03517588,
        -0.03165829,
        -0.0281407,
        -0.02462312,
        -0.02110553,
        -0.01758794,
        -0.01407035,
        -0.01055276,
        -0.00703518,
        -0.00351759,
        0.0,
    ]
)

EXPECTED_BEST_FIT_FITTING_MATH = np.array(
    [
        1.54680870e-06,
        1.55368603e-06,
        1.55991849e-06,
        1.56544892e-06,
        1.57022678e-06,
        1.57420965e-06,
        1.57736452e-06,
        1.57966886e-06,
        1.58111139e-06,
        1.58169252e-06,
        1.58142443e-06,
        1.58033076e-06,
        1.57844596e-06,
        1.57581436e-06,
        1.57248894e-06,
        1.56852991e-06,
        1.56400323e-06,
        1.55897899e-06,
        1.55352991e-06,
        1.54772985e-06,
        1.54165249e-06,
        1.53537017e-06,
        1.52895289e-06,
        1.52246757e-06,
        1.51597744e-06,
        1.50954164e-06,
        1.50321504e-06,
        1.49704814e-06,
        1.49108713e-06,
        1.48537411e-06,
        1.47994727e-06,
        1.47484130e-06,
        1.47008768e-06,
        1.46571516e-06,
        1.46175010e-06,
        1.45821700e-06,
        1.45513886e-06,
        1.45253766e-06,
        1.45043476e-06,
        1.44885132e-06,
        1.44780876e-06,
        1.44732911e-06,
        1.44743542e-06,
        1.44815218e-06,
        1.44950569e-06,
        1.45152448e-06,
        1.45423967e-06,
        1.45768541e-06,
        1.46189929e-06,
        1.46692273e-06,
        1.47280146e-06,
        1.47958590e-06,
        1.48733163e-06,
        1.49609983e-06,
        1.50595765e-06,
        1.51697860e-06,
        1.52924289e-06,
        1.54283756e-06,
        1.55785658e-06,
        1.57440065e-06,
        1.59257668e-06,
        1.61249687e-06,
        1.63427715e-06,
        1.65803480e-06,
        1.68388510e-06,
        1.71193652e-06,
        1.74228427e-06,
        1.77500175e-06,
        1.81012950e-06,
        1.84766128e-06,
        1.88752711e-06,
        1.92957334e-06,
        1.97354017e-06,
        2.01903801e-06,
        2.06552475e-06,
        2.11228738e-06,
        2.15843256e-06,
        2.20289161e-06,
        2.24444551e-06,
        2.28177427e-06,
        2.31353193e-06,
        2.33844327e-06,
        2.35541281e-06,
        2.36363074e-06,
        2.36265824e-06,
        2.35247595e-06,
        2.33348600e-06,
        2.30646702e-06,
        2.27249133e-06,
        2.23282001e-06,
        2.18879351e-06,
        2.14173347e-06,
        2.09286574e-06,
        2.04326886e-06,
        1.99384717e-06,
        1.94532441e-06,
        1.89825215e-06,
        1.85302763e-06,
        1.80991627e-06,
        1.76907529e-06,
        1.73057639e-06,
        1.69442585e-06,
        1.66058181e-06,
        1.62896837e-06,
        1.59948685e-06,
        1.57202456e-06,
        1.54646142e-06,
        1.52267481e-06,
        1.50054310e-06,
        1.47994808e-06,
        1.46077654e-06,
        1.44292130e-06,
        1.42628171e-06,
        1.41076392e-06,
        1.39628079e-06,
        1.38275179e-06,
        1.37010267e-06,
        1.35826512e-06,
        1.34717645e-06,
        1.33677913e-06,
        1.32702045e-06,
        1.31785213e-06,
        1.30922992e-06,
        1.30111334e-06,
        1.29346528e-06,
        1.28625173e-06,
        1.27944154e-06,
        1.27300611e-06,
        1.26691922e-06,
        1.26115677e-06,
        1.25569661e-06,
        1.25051838e-06,
        1.24560333e-06,
        1.24093419e-06,
        1.23649503e-06,
        1.23227115e-06,
        1.22824898e-06,
        1.22441596e-06,
        1.22076047e-06,
        1.21727177e-06,
        1.21393987e-06,
        1.21075553e-06,
    ]
)

EXPECTED_PEAK2_FITTING_MATH = np.array(
    [
        4.27919107e-08,
        4.37970519e-08,
        4.48375606e-08,
        4.59150925e-08,
        4.70313999e-08,
        4.81883383e-08,
        4.93878736e-08,
        5.06320903e-08,
        5.19231998e-08,
        5.32635495e-08,
        5.46556329e-08,
        5.61021006e-08,
        5.76057715e-08,
        5.91696464e-08,
        6.07969211e-08,
        6.24910020e-08,
        6.42555221e-08,
        6.60943595e-08,
        6.80116565e-08,
        7.00118411e-08,
        7.20996501e-08,
        7.42801550e-08,
        7.65587895e-08,
        7.89413802e-08,
        8.14341800e-08,
        8.40439047e-08,
        8.67777735e-08,
        8.96435529e-08,
        9.26496055e-08,
        9.58049435e-08,
        9.91192873e-08,
        1.02603130e-07,
        1.06267811e-07,
        1.10125589e-07,
        1.14189736e-07,
        1.18474628e-07,
        1.22995852e-07,
        1.27770322e-07,
        1.32816407e-07,
        1.38154077e-07,
        1.43805049e-07,
        1.49792971e-07,
        1.56143602e-07,
        1.62885025e-07,
        1.70047869e-07,
        1.77665566e-07,
        1.85774613e-07,
        1.94414874e-07,
        2.03629892e-07,
        2.13467230e-07,
        2.23978831e-07,
        2.35221396e-07,
        2.47256767e-07,
        2.60152322e-07,
        2.73981347e-07,
        2.88823381e-07,
        3.04764499e-07,
        3.21897494e-07,
        3.40321912e-07,
        3.60143856e-07,
        3.81475477e-07,
        4.04434023e-07,
        4.29140286e-07,
        4.55716250e-07,
        4.84281687e-07,
        5.14949396e-07,
        5.47818746e-07,
        5.82967115e-07,
        6.20438842e-07,
        6.60231338e-07,
        7.02278123e-07,
        7.46428874e-07,
        7.92426989e-07,
        8.39885921e-07,
        8.88266467e-07,
        9.36858404e-07,
        9.84771052e-07,
        1.03093826e-06,
        1.07414343e-06,
        1.11306890e-06,
        1.14637089e-06,
        1.17277631e-06,
        1.19119168e-06,
        1.20080912e-06,
        1.20119162e-06,
        1.19232161e-06,
        1.17460286e-06,
        1.14881561e-06,
        1.11603372e-06,
        1.07751969e-06,
        1.03461540e-06,
        9.88643822e-07,
        9.40832060e-07,
        8.92259873e-07,
        8.43832777e-07,
        7.96275615e-07,
        7.50141033e-07,
        7.05827292e-07,
        6.63600767e-07,
        6.23619638e-07,
        5.85956477e-07,
        5.50618443e-07,
        5.17564483e-07,
        4.86719479e-07,
        4.57985516e-07,
        4.31250628e-07,
        4.06395414e-07,
        3.83297930e-07,
        3.61837182e-07,
        3.41895565e-07,
        3.23360465e-07,
        3.06125259e-07,
        2.90089852e-07,
        2.75160894e-07,
        2.61251764e-07,
        2.48282395e-07,
        2.36179001e-07,
        2.24873734e-07,
        2.14304309e-07,
        2.04413620e-07,
        1.95149346e-07,
        1.86463578e-07,
        1.78312457e-07,
        1.70655829e-07,
        1.63456931e-07,
        1.56682092e-07,
        1.50300459e-07,
        1.44283752e-07,
        1.38606027e-07,
        1.33243473e-07,
        1.28174217e-07,
        1.23378155e-07,
        1.18836790e-07,
        1.14533095e-07,
        1.10451378e-07,
        1.06577166e-07,
        1.02897104e-07,
        9.93988496e-08,
        9.60709927e-08,
        9.29029733e-08,
        8.98850100e-08,
        8.70080348e-08,
    ]
)