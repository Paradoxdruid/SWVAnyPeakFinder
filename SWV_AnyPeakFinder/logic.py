#!/usr/bin/env python3

"""
This is the internal logic of SWV Any PeakFinder.

This program finds peak height values (i.e., peak currents) from .txt files
and .csv files containing squarewave voltammogram data, using any
selected files.
"""


import csv
import os
import re
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Type

import _csv
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters
from lmfit.models import ConstantModel, LinearModel, LorentzianModel, QuadraticModel

if TYPE_CHECKING:  # pragma: no cover
    from SWV_AnyPeakFinder.gui import PeakFinderApp


@dataclass
class FitResults:
    """Dataclass for holding fit result parameters."""

    model: str
    result: Any
    background: Any
    ip: float
    chisqr: float


class PeakLogicFiles:
    """This is the internal logic of Peak Finder.

    PeaklogicFiles looks at a user-provided list of files, then fits
    the square wave voltammogram data inside to a non-linear polynomial
    plus gaussian function, subtracts the polynomial, and reports the
    peak current.  Ultimately, it prints a pretty graph of the data."""

    def __init__(self, app: "PeakFinderApp") -> None:
        """Initialize PeakLogic, passing the gui object to this class."""

        self.app: "PeakFinderApp" = app

    def peakfinder(
        self,
    ) -> Optional[Tuple[List[int], List[float], str]]:
        """PeakLogic.peakfinder is the core function to find and report
        peak current values."""

        # Make sure the user has selected files
        if int(self.app.dir_selected.get()) != 1:  # pragma: no cover
            return None

        # grab the text variables that we need
        filename: str = str(self.app.filename_.get())
        filenamesList: List[str] = self.app.filenames_
        path: str = os.path.dirname(os.path.normpath(filenamesList[0]))
        self.app.bar.set_maxval(len(filenamesList) + 1)
        os.chdir(path)

        # open our self.output file and set it up
        with open("{}.csv".format(str(filename)), "w") as self.g:
            self.g.write("{}\n\n".format(str(filename)))
            self.g.write("Fitting Parameters\n")
            self.g.write("Peak Center,{}\n".format(str(self.app.peak_center_.get())))
            self.g.write("Left Edge,{}\n".format(str(self.app.init_edge_.get())))
            self.g.write("Right Edge,{}\n\n".format(str(self.app.final_edge_.get())))
            self.g.write("time,file,peak current\n")

            # run the peakfinder
            printing_list: List[int]
            iplist: List[float]
            printing_list, iplist = self.loop_for_peaks_files(filenamesList)

        # Catch if peakfinder failed
        if not all([printing_list, iplist]):
            self.app.output.set("Program failed!")
            return None

        # Otherwise, show the user what was found
        # self.app.output.set("Wrote output to {}.csv".format(filename))
        # mainGraph =
        peak_output: Tuple[List[int], List[float], str] = (
            printing_list,
            iplist,
            filename,
        )
        # self.app.output.set(f"Returning {len(peak_output)} objects")
        return peak_output

    def loop_for_peaks_files(
        self, filenamesList: List[str]
    ) -> Tuple[List[int], List[float]]:
        """PeakLogic.loopForPeaks() will loop through each file,
        collecting data and sending it to the peak_math function."""

        # clear some lists to hold our data
        full_x_lists: List[List[str]] = []
        full_y_lists: List[List[str]] = []
        startT: int = -1
        timelist: List[int] = []
        printing_list: List[str] = []

        plt.close(2)  # close test fitting graph if open

        for each in filenamesList:  # loop through each file
            try:
                dialect: Type[csv.Dialect] = csv.Sniffer().sniff(
                    open(each).read(1024), delimiters="\t,"
                )
                open(each).seek(0)
                self.f: _csv._reader = csv.reader(open(each), dialect)
                listfile: List[List[str]] = list(self.f)
                t_list: List[int] = []
                y_list: List[str] = []
                rx_list: List[str] = []

                # remove the header rows from the file, leaving just the data
                start_pattern: int = 3
                for index, line in enumerate(listfile):
                    # try:
                    if line[0]:
                        if re.match("Potential*", line[0]):
                            start_pattern = index
                    # except Exception:
                    #     pass
                datalist: List[List[str]] = listfile[start_pattern + 2 :]
                pointT: int = 1000
                # if it's the first data point, set the initial time to zero
                if startT == -1:
                    startT = pointT
                # subtract init time to get time since the start of the trial
                pointTcorr: int = pointT - startT
                t_list.append(pointTcorr)
                for row in datalist:
                    if row == []:  # skip empty lines
                        pass
                    else:
                        if row[0]:
                            rx_list.append(row[0])
                        if row[1]:
                            y_list.append(row[1])
                        else:
                            pass
                full_x_lists.append(rx_list)
                full_y_lists.append(y_list)
                timelist.append(pointTcorr)
                justName: str = os.path.split(each)[1]
                printing_list.append(justName)
            except IndexError:  # Does this exception catch everything?
                pass
        iplist: List[float] = self.peak_math(full_x_lists, full_y_lists)

        # write the output csv file
        for i, v, y in zip(iplist, timelist, printing_list):
            self.g.write("{0},{1},{2}\n".format(str(v), str(y), str(i)))

        # return time and peak current for graphing
        return timelist, iplist

    def peak_math(
        self, listsx: List[List[str]], listsy: List[List[str]]
    ) -> List[float]:
        """PeakLogic.peak_math() passes each data file to .fitting_math,
        and returns a list of peak currents."""

        iplist: List[float] = []
        count: int = 1

        for xfile, yfile in zip(listsx, listsy):
            ip = self.fitting_math(xfile, yfile, 1)

            # check data quality
            if ip < 0:  # pragma: no cover
                ip = 0
            iplist.append(ip)
            self.app.bar.update(count)
            count = count + 1

        return iplist

    @staticmethod
    def add_lz_peak(
        prefix: str, center: float, amplitude: float = 0.000005, sigma: float = 0.05
    ) -> Tuple[LorentzianModel, Any]:
        """Generate a reasonable Lorenzian peak for model fitting."""
        peak = LorentzianModel(prefix=prefix)
        pars = peak.make_params()
        pars[prefix + "center"].set(center)
        pars[prefix + "amplitude"].set(amplitude, min=0)
        pars[prefix + "sigma"].set(sigma, min=0)
        return peak, pars

    @staticmethod
    def get_r2(
        x: "np.ndarray[Any, np.dtype[np.float64]]",
        y: "np.ndarray[Any, np.dtype[np.float64]]",
    ) -> "np.ndarray[Any, np.dtype[np.float64]]":
        return np.corrcoef(x, y)[0, 1] ** 2

    def fitting_math(
        self,
        xfile: List[str],
        yfile: List[str],
        flag: int = 1,
    ) -> Any:
        """PeakLogic.fitting_math() fits the data to linear background
        with a water and methylene blue peak, then subtracts water peak and
        linear slope to find peak current.."""

        try:
            center: float = self.app.peak_center_.get()
            x: "np.ndarray[Any, np.dtype[np.float64]]" = np.array(
                xfile, dtype=np.float64
            )
            y: "np.ndarray[Any, np.dtype[np.float64]]" = np.array(
                yfile, dtype=np.float64
            )

            # cut out outliers
            passingx: "np.ndarray[Any, np.dtype[np.float64]]"
            passingy: "np.ndarray[Any, np.dtype[np.float64]]"
            passingx, passingy = self.trunc_edges(xfile, yfile)

            # Test different models
            # https://lmfit-py.readthedocs.io/en/0.9.0/builtin_models.html

            model = self._return_best_model(passingx, passingy, center)

            if flag == 1:
                return model.ip
            if flag == 0:
                return (
                    x,
                    y,
                    model.result.best_fit,
                    model.background,
                    model.ip,
                    passingx,
                )

        except Exception:  # pragma: no cover
            print("Error Fitting")
            print(sys.exc_info())
            return -1

    def _return_best_model(
        self,
        x: "np.ndarray[Any, np.dtype[np.float64]]",
        y: "np.ndarray[Any, np.dtype[np.float64]]",
        center: float,
    ) -> FitResults:
        """Evaluate multiple models for peak fitting and return best fit model."""

        models = [
            self._quadratic_model,
            self._one_peak_model,
            self._two_peak_model,
            # self._three_peak_model,
        ]

        outcomes = [each(x, y, center) for each in models]

        # Find minimum chisqr and return that model
        best_model: FitResults = min(outcomes, key=lambda x: x.chisqr)

        return best_model

    @staticmethod
    def _create_linear_background() -> Tuple[LinearModel, Parameters]:
        model = LinearModel(prefix="Background")
        params = model.make_params()
        params.add("Backgroundslope", 0)  # , min=0)
        params.add("Backgroundintercept", 0, min=0)

        return model, params

    @staticmethod
    def _find_background_regions(
        x: "np.ndarray[Any, np.dtype[np.float64]]",
        y: "np.ndarray[Any, np.dtype[np.float64]]",
        pleft: float,
        pright: float,
    ) -> Tuple[List[float], List[float]]:
        bkg_x_left = [i for i in x if i < pleft]
        if len(bkg_x_left) == 0:
            bkg_x_left = list(x[:10])
        bkg_x_right = [i for i in x if i > pright]
        if len(bkg_x_right) == 0:
            bkg_x_right = list(x[-10:])
        bkg_x = list(bkg_x_left) + list(bkg_x_right)
        bkg_y = list(y[: len(bkg_x_left)]) + list(y[-len(bkg_x_right) :])
        return bkg_x, bkg_y

    def _one_peak_model(
        self,
        x: "np.ndarray[Any, np.dtype[np.float64]]",
        y: "np.ndarray[Any, np.dtype[np.float64]]",
        center: float,
    ) -> Any:
        rough_peak_positions = [center]

        model, params = self._create_linear_background()

        for i, cen in enumerate(rough_peak_positions):
            peak, pars = self.add_lz_peak(f"Peak_{i+1}", cen, (max(y) - min(y)))
            model = model + peak
            params.update(pars)

        result = model.fit(y, params, x=x)

        # Now, correct for edge affects
        pcenter = result.values["Peak_1center"]
        psigma = result.values["Peak_1sigma"]
        pleft = pcenter - (2 * psigma)
        pright = pcenter + (2 * psigma)

        bkg_x, bkg_y = self._find_background_regions(x, y, pleft, pright)

        rough_peak_positions = [pcenter]
        edge_model = LinearModel(prefix="Background")
        edge_params = edge_model.make_params()
        edge_result = edge_model.fit(data=bkg_y, params=edge_params, x=bkg_x)

        # Refit with fixed background
        rough_peak_positions = [pcenter]
        final_model = LinearModel(prefix="Background")
        final_params = final_model.make_params()
        final_params.add(
            "Backgroundslope", edge_result.best_values["Backgroundslope"], vary=False
        )
        final_params.add(
            "Backgroundintercept",
            edge_result.best_values["Backgroundintercept"],
            vary=False,
        )

        for i, cen in enumerate(rough_peak_positions):
            peak, pars = self.add_lz_peak(f"Peak_{i+1}", center)
            final_model = final_model + peak
            final_params.update(pars)

        final_result = final_model.fit(y, final_params, x=x)
        final_comps = final_result.eval_components()

        ip = float(max(final_comps["Peak_1"]))

        model = FitResults(
            "linear", final_result, final_comps["Background"], ip, final_result.chisqr
        )

        return model

    def _two_peak_model(
        self,
        x: "np.ndarray[Any, np.dtype[np.float64]]",
        y: "np.ndarray[Any, np.dtype[np.float64]]",
        center: float,
    ) -> Any:
        rough_peak_positions = [min(x), center]

        model, params = self._create_linear_background()

        for i, cen in enumerate(rough_peak_positions):
            peak, pars = self.add_lz_peak(f"Peak_{i+1}", cen, (max(y) - min(y)))
            model = model + peak
            params.update(pars)

        result = model.fit(y, params, x=x)

        # Now, correct for edge affects
        pcenter = result.values["Peak_2center"]
        psigma = result.values["Peak_2sigma"]
        pleft = pcenter - (2 * psigma)
        pright = pcenter + (2 * psigma)

        bkg_x, bkg_y = self._find_background_regions(x, y, pleft, pright)

        rough_peak_positions = [min(bkg_x)]
        edge_model = LinearModel(prefix="Background") + LorentzianModel(prefix="Peak_1")
        edge_params = edge_model.make_params()

        edge_result = edge_model.fit(data=bkg_y, params=edge_params, x=bkg_x)

        # Refit with fixed background
        final_model = LinearModel(prefix="Background") + LorentzianModel(
            prefix="Peak_1"
        )
        final_params = final_model.make_params()
        final_params.add(
            "Backgroundslope", edge_result.best_values["Backgroundslope"], vary=False
        )
        final_params.add(
            "Backgroundintercept",
            edge_result.best_values["Backgroundintercept"],
            vary=False,
        )
        final_params.add(
            "Peak_1center", edge_result.best_values["Peak_1center"], vary=False
        )
        final_params.add(
            "Peak_1amplitude", edge_result.best_values["Peak_1amplitude"], vary=False
        )
        final_params.add(
            "Peak_1sigma", edge_result.best_values["Peak_1sigma"], vary=False
        )

        peak, pars = self.add_lz_peak("Peak_2", pcenter)
        final_model = final_model + peak
        final_params.update(pars)

        final_result = final_model.fit(y, final_params, x=x)
        final_comps = final_result.eval_components()

        ip = float(max(final_comps["Peak_2"]))
        background = final_comps["Background"] + final_comps["Peak_1"]

        model = FitResults("one shoulder", result, background, ip, final_result.chisqr)

        return model

    def _quadratic_model(
        self,
        x: "np.ndarray[Any, np.dtype[np.float64]]",
        y: "np.ndarray[Any, np.dtype[np.float64]]",
        center: float,
    ) -> Any:
        rough_peak_positions = [center]

        min_y = float(min(y))
        model = QuadraticModel(prefix="Background") + ConstantModel(prefix="Baseline")
        params = model.make_params()
        params.add("Backgrounda", 0.000001, min=0)
        params.add("Backgroundb", 0.000001, min=0)
        params.add("Backgroundc", 0.000005, min=max([0, min_y]))
        params.add("Baselinec", 0, min=0)

        for i, cen in enumerate(rough_peak_positions):
            peak, pars = self.add_lz_peak(f"Peak_{i+1}", cen, (max(y) - min(y)))
            model = model + peak
            params.update(pars)

        _ = model.eval(params, x=x)
        result = model.fit(y, params, x=x)

        # Now, correct for edge affects
        pcenter = result.values["Peak_1center"]
        psigma = result.values["Peak_1sigma"]
        pleft = pcenter - (2 * psigma)
        pright = pcenter + (2 * psigma)

        bkg_x, bkg_y = self._find_background_regions(x, y, pleft, pright)

        rough_peak_positions = [pcenter]
        edge_model = QuadraticModel(prefix="Background") + ConstantModel(
            prefix="Baseline"
        )
        edge_params = edge_model.make_params()

        edge_result = edge_model.fit(data=bkg_y, params=edge_params, x=bkg_x)

        # Refit with fixed background
        rough_peak_positions = [pcenter]
        final_model = QuadraticModel(prefix="Background") + ConstantModel(
            prefix="Baseline"
        )
        final_params = final_model.make_params()
        final_params.add(
            "Backgrounda", edge_result.best_values["Backgrounda"], vary=False
        )
        final_params.add(
            "Backgroundb", edge_result.best_values["Backgroundb"], vary=False
        )
        final_params.add(
            "Backgroundc", edge_result.best_values["Backgroundc"], vary=False
        )
        final_params.add("Baselinec", edge_result.best_values["Baselinec"], vary=False)

        for i, cen in enumerate(rough_peak_positions):
            peak, pars = self.add_lz_peak(f"Peak_{i+1}", center)
            final_model = final_model + peak
            final_params.update(pars)

        final_result = final_model.fit(y, final_params, x=x)
        final_comps = final_result.eval_components()

        ip = float(max(final_comps["Peak_1"]))
        background = final_comps["Background"] + final_comps["Baseline"]

        model = FitResults("quadratic", result, background, ip, final_result.chisqr)

        return model

    def _three_peak_model(
        self,
        x: "np.ndarray[Any, np.dtype[np.float64]]",
        y: "np.ndarray[Any, np.dtype[np.float64]]",
        center: float,
    ) -> Any:
        rough_peak_positions = [min(x), center, max(x)]

        model, params = self._create_linear_background()

        for i, cen in enumerate(rough_peak_positions):
            peak, pars = self.add_lz_peak(f"Peak_{i+1}", cen, (max(y) - min(y)))
            model = model + peak
            params.update(pars)

        result = model.fit(y, params, x=x)

        # Now, correct for edge affects
        pcenter = result.values["Peak_2center"]
        psigma = result.values["Peak_2sigma"]
        pleft = pcenter - (2 * psigma)
        pright = pcenter + (2 * psigma)

        bkg_x, bkg_y = self._find_background_regions(x, y, pleft, pright)

        rough_peak_positions = [min(bkg_x), max(bkg_x)]
        edge_model = (
            LinearModel(prefix="Background")
            + LorentzianModel(prefix="Peak_1")
            + LorentzianModel(prefix="Peak_3")
        )
        edge_params = edge_model.make_params()
        edge_params.add("Peak_1center", result.best_values["Peak_1center"])
        edge_params.add("Peak_1amplitude", result.best_values["Peak_1amplitude"])
        edge_params.add("Peak_1sigma", result.best_values["Peak_1sigma"])
        edge_params.add("Peak_3center", result.best_values["Peak_3center"])
        edge_params.add("Peak_3amplitude", result.best_values["Peak_3amplitude"])
        edge_params.add("Peak_3sigma", result.best_values["Peak_3sigma"])

        edge_result = edge_model.fit(data=bkg_y, params=edge_params, x=bkg_x)

        # Refit with fixed background
        final_model = (
            LinearModel(prefix="Background")
            + LorentzianModel(prefix="Peak_1")
            + LorentzianModel(prefix="Peak_3")
        )
        final_params = final_model.make_params()
        final_params.add(
            "Backgroundslope", edge_result.best_values["Backgroundslope"], vary=False
        )
        final_params.add(
            "Backgroundintercept",
            edge_result.best_values["Backgroundintercept"],
            vary=False,
        )
        final_params.add(
            "Peak_1center", edge_result.best_values["Peak_1center"], vary=False
        )
        final_params.add(
            "Peak_1amplitude", edge_result.best_values["Peak_1amplitude"], vary=False
        )
        final_params.add(
            "Peak_1sigma", edge_result.best_values["Peak_1sigma"], vary=False
        )
        final_params.add(
            "Peak_3center", edge_result.best_values["Peak_3center"], vary=False
        )
        final_params.add(
            "Peak_3amplitude", edge_result.best_values["Peak_3amplitude"], vary=False
        )
        final_params.add(
            "Peak_3sigma", edge_result.best_values["Peak_3sigma"], vary=False
        )

        peak, pars = self.add_lz_peak("Peak_2", pcenter)
        final_model = final_model + peak
        final_params.update(pars)

        final_result = final_model.fit(y, final_params, x=x)
        final_comps = final_result.eval_components()

        ip = float(max(final_comps["Peak_2"]))
        background = (
            final_comps["Background"] + final_comps["Peak_1"] + final_comps["Peak_3"]
        )

        model = FitResults("two shoulder", result, background, ip, final_result.chisqr)

        return model

    def trunc_edges(
        self, listx: List[str], listy: List[str]
    ) -> Tuple[
        "np.ndarray[Any, np.dtype[np.float64]]", "np.ndarray[Any, np.dtype[np.float64]]"
    ]:
        """PeakLogic.trunc_edges() removes outlier regions of known
        bad signal from an x-y data list and returns the inner edges."""

        newx: List[float] = []
        newy: List[float] = []
        start_spot: str
        start_h: str
        for start_spot, start_h in zip(listx, listy):
            spot: float = float(start_spot)
            h: float = float(start_h)
            low: float = float(self.app.final_edge_.get())
            high: float = float(self.app.init_edge_.get())
            if spot > low:  # add low values
                if spot < high:
                    newx.append(spot)
                    newy.append(h)
                else:  # pragma: no cover
                    pass
            else:
                pass

        # convert results back to an array
        px: "np.ndarray[Any, np.dtype[np.float64]]" = np.array(newx, dtype=np.float64)
        py: "np.ndarray[Any, np.dtype[np.float64]]" = np.array(newy, dtype=np.float64)
        return px, py

    def test_fit(self, dataind: int = 0) -> None:
        """Perform a fit for the first data point and display it for
        the user."""

        # Make sure the user has selected a directory
        if int(self.app.dir_selected.get()) == 1:
            try:
                filenamesList: List[str] = self.app.filenames_
                file: str = filenamesList[dataind]
                dialect: Type[csv.Dialect] = csv.Sniffer().sniff(
                    open(file).read(1024), delimiters="\t,"
                )

                open(file).seek(0)  # open the first data file
                self.testfile: _csv._reader = csv.reader(open(file), dialect)
                listfile: List[List[str]] = list(self.testfile)

                # remove the header rows from the file, leaving just the data
                start_pattern: int = 3
                for index, line in enumerate(listfile):
                    try:
                        if line[0]:
                            if re.match("Potential*", line[0]):
                                start_pattern = index
                    except IndexError:
                        pass
                datalist: List[List[str]] = listfile[start_pattern + 2 :]

                x_list: List[str] = []
                y_list: List[str] = []
                for row in datalist:
                    if row == []:  # skip empty lines
                        pass
                    else:
                        if row[0]:
                            x_list.append(row[0])
                        if row[1]:
                            y_list.append(row[1])
                        else:
                            pass

                x: "np.ndarray[Any, np.dtype[np.float64]]"
                y: "np.ndarray[Any, np.dtype[np.float64]]"
                y_fp: "np.ndarray[Any, np.dtype[np.float64]]"
                y_bkg: "np.ndarray[Any, np.dtype[np.float64]]"
                ip: float
                px: "np.ndarray[Any, np.dtype[np.float64]]"

                x, y, y_fp, y_bkg, ip, px = self.fitting_math(x_list, y_list, flag=0)
                self.test_grapher(x, y, y_fp, y_bkg, file, ip, px)

            except (ValueError, IndexError):
                pass
        else:
            pass

    def test_grapher(
        self,
        x: "np.ndarray[Any, np.dtype[np.float64]]",
        y: "np.ndarray[Any, np.dtype[np.float64]]",
        y_fp: "np.ndarray[Any, np.dtype[np.float64]]",
        y_bkg: "np.ndarray[Any, np.dtype[np.float64]]",
        file: str,
        ip: float,
        px: "np.ndarray[Any, np.dtype[np.float64]]",
    ) -> None:  # pragma: no cover
        """PeakLogic.test_grapher() displays a graph of the test fitting."""

        plt.close(2)  # close previous test if open

        # add background components
        full_bkg: "np.ndarray[Any, np.dtype[np.float64]]" = y_bkg

        file_name: str = os.path.basename(file)
        self.fig2 = plt.figure(2)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.plot(x, y, "r.", label="data")
        self.ax2.plot(px, y_fp, label="fit")
        self.ax2.plot(px, full_bkg, label="background")
        self.ax2.set_xlabel("Potential (V)")
        self.ax2.set_ylabel("Current (A)")
        self.ax2.set_title("Fit of {}".format(str(file_name)))
        self.ax2.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")
        self.ax2.legend()
        self.fig2.subplots_adjust(bottom=0.15)
        self.fig2.subplots_adjust(left=0.15)
        self.text = self.ax2.text(
            0.1,
            0.95,
            "Peak Current:\n%.2e A" % ip,
            transform=self.ax2.transAxes,
            va="top",
        )

        self.fig2.canvas.draw()
        plt.show()
