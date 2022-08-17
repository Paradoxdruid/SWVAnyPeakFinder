"""Unit tests for SWV_AnyPeakFinder.logic"""

import numpy as np
import SWV_AnyPeakFinder.gui as gui
import SWV_AnyPeakFinder.logic as logic
from lmfit.models import LorentzianModel

import test_values


def setup_PeakFinderApp(mocker):
    mocker.patch("SWV_AnyPeakFinder.gui.PeakFinderApp")
    app = gui.PeakFinderApp()
    app.peak_center_.get.return_value = -0.4
    app.final_edge_.get.return_value = -0.6
    app.init_edge_.get.return_value = -0.1
    logic2 = logic.PeakLogicFiles(app)

    return logic2


def test_PeakLogicFiles_trunc_edges(mocker):
    mocker.patch("SWV_AnyPeakFinder.gui.PeakFinderApp")
    app = gui.PeakFinderApp()
    app.final_edge_.get.return_value = -0.9
    app.init_edge_.get.return_value = -0.5

    TEST_X = np.array([-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4])
    TEST_Y = np.array([1, 2, 3, 4, 5, 6, 7])
    EXPECTED = (np.array([-0.8, -0.7, -0.6]), np.array([3, 4, 5]))
    logic2 = logic.PeakLogicFiles(app)
    actual = logic2.trunc_edges(TEST_X, TEST_Y)

    np.testing.assert_array_equal(EXPECTED, actual)
    # assert (EXPECTED == actual).all()


def test_PeakLogicFiles_add_lz_peak(mocker):
    TEST_PREFIX = "test"
    TEST_CENTER = -0.4
    EXPECTED_PEAK = LorentzianModel(prefix="test")
    EXPECTED_PARAMS = EXPECTED_PEAK.make_params(
        center=-0.4, amplitude=0.005, sigma=0.05
    )

    mocker.patch("SWV_AnyPeakFinder.gui.PeakFinderApp")
    app = gui.PeakFinderApp()
    logic2 = logic.PeakLogicFiles(app)

    _, actual_params = logic2.add_lz_peak(TEST_PREFIX, TEST_CENTER)
    # assert EXPECTED_PEAK == actual_peak  # FIXME
    assert EXPECTED_PARAMS == actual_params


def test_PeakLogicFiles_fitting_math_flag_0(mocker):
    logic = setup_PeakFinderApp(mocker)
    TEST_X_FILE = list(np.linspace(-0.7, 0, 200))
    x = np.linspace(-0.7, 0, 200)
    TEST_Y_FILE = list(test_values.EXPECTED_Y_ARRAY)
    TEST_FLAG = 0

    EXPECTED_X = test_values.EXPECTED_X_FITTING_MATH
    EXPECTED_Y = test_values.EXPECTED_Y_FITTING_MATH
    EXPECTED_BEST_FIT = test_values.EXPECTED_BEST_FIT_FITTING_MATH
    EXPECTED_IP = 1.1715001756255807e-06
    EXPECTED_PEAK2 = test_values.EXPECTED_PEAK2_FITTING_MATH

    x, y, best_fit, _, _, peak2, ip, _ = logic.fitting_math(
        TEST_X_FILE, TEST_Y_FILE, flag=TEST_FLAG
    )

    assert EXPECTED_IP == ip
    np.testing.assert_array_almost_equal(EXPECTED_Y, y)
    np.testing.assert_array_almost_equal(EXPECTED_X, x)
    np.testing.assert_array_almost_equal(EXPECTED_BEST_FIT, best_fit)
    np.testing.assert_array_almost_equal(EXPECTED_PEAK2, peak2)


def test_PeakLogicFiles_fitting_math_flag_1(mocker):
    logic = setup_PeakFinderApp(mocker)
    TEST_X_FILE = list(np.linspace(-0.7, 0, 200))
    TEST_Y_FILE = list(test_values.EXPECTED_Y_ARRAY)
    TEST_FLAG = 1

    EXPECTED_IP = 1.1715001756255807e-06

    ip = logic.fitting_math(TEST_X_FILE, TEST_Y_FILE, flag=TEST_FLAG)

    assert EXPECTED_IP == ip


def test_PeakLogicFiles_test_fit(mocker):
    pass
    # mocker.patch("SWV_AnyPeakFinder.SWV_AnyPeakFinder.PeakFinderApp")
    # app = swv.PeakFinderApp()
    # app.peak_center_.get.return_value = -0.4
    # app.final_edge_.get.return_value = -0.6
    # app.init_edge_.get.return_value = -0.1
    # app.dir_selected.get.return_value = 1

    # app.filenames_.return_value = ["Test1.csv"]

    # mocker.patch("csv.Sniffer.sniff")
    # mocker.patch("SWV_AnyPeakFinder.SWV_AnyPeakFinder.PeakLogicFiles.test_grapher")
    # logic = swv.PeakLogicFiles(app)

    # logic.test_fit()


def test_PeakLogicFiles_peak_math(mocker):
    logic = setup_PeakFinderApp(mocker)
    TEST_X_FILE = [(list(np.linspace(-0.7, 0, 200)))]
    TEST_Y_FILE = [(list(test_values.EXPECTED_Y_ARRAY))]

    EXPECTED_IP_LIST = [1.1715001756255807e-06]
    iplist = logic.peak_math(TEST_X_FILE, TEST_Y_FILE)

    assert EXPECTED_IP_LIST == iplist
