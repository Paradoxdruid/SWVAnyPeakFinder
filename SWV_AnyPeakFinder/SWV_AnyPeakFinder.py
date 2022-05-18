#!/usr/bin/env python3

"""
SWV Any PeakFinder.

This program finds peak height values (i.e., peak currents) from .txt files
and .csv files containing squarewave voltammogram data, using any
selected files.
"""


import csv
import os
import platform
import re
import sys
import tkinter
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any, List, Tuple, Type

import _csv
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import LinearModel, LorentzianModel

import SWV_AnyPeakFinder.__version__ as version

if platform.system() == "Darwin":  # pragma: no cover
    import matplotlib

    matplotlib.use("TkAgg")


class PointBrowser:
    """This is the class that draws the main graph of data for Peak Finder.

    This class creates a line and point graph with clickable points.
    When a point is clicked, it calls the normal test Fit display
    from the main logic."""

    def __init__(
        self,
        app: "PeakFinderApp",
        logic: "PeakLogicFiles",
        xticksRaw: List[int],
        y: List[float],
        fileTitle: str,
    ) -> None:
        """Create the main output graph and make it clickable."""

        self.app: "PeakFinderApp" = app
        self.logic: "PeakLogicFiles" = logic
        self.x: List[float] = list(range(len(xticksRaw)))
        self.y: List[float] = y
        self.xticks: Tuple[float, ...] = tuple(xticksRaw)
        self.loc: List[float] = [d + 0.5 for d in self.x]

        # Setup the matplotlib figure
        self.fig = plt.figure(1)
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(self.x, self.y, "bo-", picker=True, pickradius=5)
        self.fig.canvas.mpl_connect("pick_event", self.onpick)
        self.fig.subplots_adjust(left=0.17)
        self.ax.set_xlabel("File")
        self.ax.get_xaxis().set_ticks([])
        self.ax.set_ylabel("Current (A)")
        self.ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")
        self.ax.set_title("Peak Current for {}".format(str(fileTitle)))

        # Track which point is selected by the user
        self.lastind: int = 0

        # Add our data
        self.selected = self.ax.plot(
            [self.x[0]],
            [self.y[0]],
            "o",
            ms=12,
            alpha=0.4,
            color="yellow",
            visible=True,
        )

        # Display button to fetch data
        self.axb = plt.axes([0.75, 0.03, 0.15, 0.05])
        self.button = plt.Button(self.axb, "Results")
        # self.ax.plot._test = self.button
        # self.button.on_clicked(self.app.data_popup)

        # Draw the figure
        self.fig.canvas.draw()
        plt.show()

    def onpick(self, event: Any) -> None:
        """Capture the click event, find the corresponding data
        point, then update accordingly."""

        thisone = event.artist
        x = thisone.get_xdata()
        ind = event.ind

        dataind = x[ind[0]]

        self.logic.test_fit(dataind)

        self.fig.canvas.draw()
        plt.show()

    def update(self) -> None:
        """Update the main graph and call my response function."""

        if self.lastind is None:
            return
        dataind: int = self.lastind

        self.selected.set_visible(True)
        self.selected.set_data(self.x[dataind], self.y[dataind])

        self.logic.test_fit(dataind)

        self.fig.canvas.draw()
        plt.show()


class PeakFinderApp(tkinter.Tk):  # pragma: no cover
    """This is the gui for Peak Finder.

    This application displays a minimal user interface to select a
    directory and to specify file formats and output filename, then
    reports on the programs function with a simple progressbar."""

    def __init__(self) -> None:
        """set up the peak finder app GUI."""

        self.directory_manager()
        tkinter.Tk.__init__(self)
        self.window_title: str = "SWV AnyPeakFinder {}".format(str(version.__version__))

        # invoke PeakLogicFiles to do the actual work
        logic: "PeakLogicFiles" = PeakLogicFiles(self)

        # create the frame for the main window
        self.title(self.window_title)
        self.geometry("+100+100")
        self.resizable(tkinter.FALSE, tkinter.FALSE)
        mainframe: ttk.Frame = ttk.Frame(self)
        mainframe.grid(
            column=0,
            row=0,
            sticky="NWES",  # (tkinter.N, tkinter.W, tkinter.E, tkinter.S),
            padx=5,
            pady=5,
        )
        mainframe.columnconfigure(0, weight=1)  # type: ignore
        mainframe.rowconfigure(0, weight=1)  # type: ignore

        # define our variables in tkinter-form and give sane defaults
        self.filename_: tkinter.StringVar = tkinter.StringVar(value="output1")
        self.output: tkinter.StringVar = tkinter.StringVar()
        self.filenames_: List[str] = []
        self.dir_selected: tkinter.IntVar = tkinter.IntVar(value=0)
        self.init_potential_: tkinter.DoubleVar = tkinter.DoubleVar(value=-0.2)
        self.final_potential_: tkinter.DoubleVar = tkinter.DoubleVar(value=-0.4)
        self.peak_center_: tkinter.DoubleVar = tkinter.DoubleVar(value=-0.4)
        self.final_edge_: tkinter.DoubleVar = tkinter.DoubleVar(value=-1)
        self.init_edge_: tkinter.DoubleVar = tkinter.DoubleVar(value=-0.1)
        self.guesses_: None = None

        # display the entry boxes
        ttk.Label(mainframe, text=self.window_title, font="helvetica 12 bold").grid(
            column=1, row=1, sticky="W", columnspan=2
        )

        ttk.Entry(mainframe, width=12, textvariable=self.filename_).grid(
            column=2, row=2, sticky="WE"  # (tkinter.W, tkinter.E)
        )
        ttk.Label(mainframe, text="Filename:").grid(column=1, row=2, sticky="W")

        ttk.Entry(mainframe, width=12, textvariable=self.peak_center_).grid(
            column=2, row=9, sticky="WE"  # (tkinter.W, tkinter.E)
        )
        ttk.Label(mainframe, text="Peak Center:").grid(
            column=1, row=9, sticky="W"  # tkinter.W
        )

        ttk.Label(mainframe, text="Peak Location Guess", font="helvetica 10 bold").grid(
            column=1, row=5, sticky="W"  # tkinter.W
        )
        ttk.Label(mainframe, text="(only change if necessary)").grid(
            column=1, row=6, sticky="W"  # tkinter.W
        )

        # Display Generate button
        ttk.Button(mainframe, text="Find Peaks", command=logic.peakfinder).grid(
            column=1, row=10, sticky="W"
        )

        # Display test fit button
        ttk.Button(mainframe, text="Test fit", command=logic.test_fit).grid(
            column=2, row=10, sticky="W"
        )

        # Display Directory button
        ttk.Button(mainframe, text="Select Files", command=self.files_select).grid(
            column=1, row=4, sticky="W"
        )

        # Show the output
        ttk.Label(mainframe, textvariable=self.output).grid(
            column=1, row=13, sticky="WE", columnspan=5
        )

        # Set up Outlier correction
        ttk.Label(mainframe, text="Regions to include (from -> to):").grid(
            column=1, row=12, sticky="W"
        )
        ttk.Entry(mainframe, width=12, textvariable=self.final_edge_).grid(
            column=1, row=13, sticky="E"
        )
        ttk.Entry(mainframe, width=12, textvariable=self.init_edge_).grid(
            column=2, row=13, sticky="WE"
        )

        # Pad the windows for prettiness
        for child in mainframe.winfo_children():
            child.grid_configure(padx=5, pady=5)

        # Display a progressbar
        self.bar: "ProgressBar" = ProgressBar(mainframe, width=100, height=10)
        self.bar.update(0)

        ttk.Label(mainframe, text="").grid(column=1, row=16, sticky="WE")

        # add a system menu
        self.option_add("*tearOff", tkinter.FALSE)

        self.menubar: tkinter.Menu = tkinter.Menu(self)
        self["menu"] = self.menubar

        self.menu_file: tkinter.Menu = tkinter.Menu(self.menubar)
        self.menubar.add_cascade(menu=self.menu_file, label="File")

        self.menu_file.add_command(label="Exit", command=self.destroy)

        self.menu_help: tkinter.Menu = tkinter.Menu(self.menubar, name="help")
        self.menubar.add_cascade(menu=self.menu_help, label="Help")

        self.menu_help.add_command(label="How to Use", command=self.help_popup)
        self.menu_help.add_command(
            label="About Peak Finder...", command=self.about_popup
        )

        # Run the program
        plt.show()
        self.mainloop()

    def directory_manager(self) -> None:
        """Set the initial directory to the users home directory."""

        self.mydocs: str = str(Path.home())
        os.chdir(self.mydocs)

    def about_popup(self) -> None:
        """Display a pop-up menu about the program."""

        self.aboutDialog = tkinter.Toplevel(self)
        self.aboutDialog.resizable(tkinter.FALSE, tkinter.FALSE)
        self.aboutDialog.geometry("+400+100")

        aboutframe = ttk.Frame(self.aboutDialog, width="200", height="200")
        aboutframe.grid_propagate(False)
        aboutframe.grid(
            column=0,
            row=0,
            sticky="NWES",
            padx=5,
            pady=5,
        )
        aboutframe.columnconfigure(0, weight=1)  # type: ignore
        aboutframe.rowconfigure(0, weight=1)  # type: ignore
        ttk.Label(
            aboutframe,
            text=self.window_title,
            font="helvetica 12 bold",
            anchor="center",
        ).grid(column=0, row=0, sticky="NWE")
        ttk.Label(
            aboutframe,
            text=(
                "Voltammogram data analysis\nsoftware for "
                "CH Instruments data.\n\nWritten by\n{0}\n"
                "http://www.andrewjbonham.com\n{1}\n\n\n"
            ).format(str(version.__author__), str(version.__copyright__)),
            anchor="center",
            justify="center",
        ).grid(column=0, row=1, sticky="N")
        ttk.Button(aboutframe, text="Close", command=self.aboutDialog.destroy).grid(
            column=0, row=4, sticky="S"
        )
        self.aboutDialog.mainloop()

    def help_popup(self) -> None:
        """Display a pop-up menu explaining how to use the program."""

        self.helpDialog = tkinter.Toplevel(self)
        self.helpDialog.resizable(tkinter.FALSE, tkinter.FALSE)
        self.helpDialog.geometry("+400+100")
        helpframe = ttk.Frame(self.helpDialog, width="200", height="240")
        helpframe.grid(
            column=0,
            row=0,
            sticky="NWES",
            padx=5,
            pady=5,
        )
        helpframe.columnconfigure(0, weight=1)  # type: ignore
        helpframe.rowconfigure(0, weight=1)  # type: ignore
        ttk.Label(
            helpframe,
            text=("{} Help".format(self.window_title)),
            font="helvetica 12 bold",
            anchor="center",
        ).grid(column=0, row=0, sticky="NWE")
        helptext = tkinter.Text(helpframe, width=40, height=9)
        helpmessage = (
            "Peak Finder is used to find the peak current for a "
            "methylene blue reduction peak for CH instruments voltammogram "
            "data files. The data files must be sequentially numbered "
            "in the specified directory and be in text format.\n\n"
            "The initial and final potential should be adjusted to lie "
            "outside the gaussian peak."
        )
        helptext.insert("1.0", helpmessage)
        helptext.config(
            state="disabled",
            bg="SystemButtonFace",
            borderwidth=0,
            font="helvetica 10",
            wrap="word",
        )
        helptext.grid(column=0, row=1, sticky=tkinter.N, padx=10)
        ttk.Button(helpframe, text="Close", command=self.helpDialog.destroy).grid(
            column=0, row=4, sticky=(tkinter.S)
        )
        self.helpDialog.mainloop()

    def data_popup(self, event: Any) -> None:
        """Display a pop-up window of data."""

        filename: str = "{}.csv".format(str(self.filename_.get()))
        self.dataDialog = tkinter.Toplevel(self)
        self.dataDialog.resizable(tkinter.FALSE, tkinter.FALSE)
        self.dataDialog.geometry("+400+100")
        dataframe = ttk.Frame(self.dataDialog, width="300", height="600")
        dataframe.grid_propagate(False)
        dataframe.grid(
            column=0,
            row=0,
            sticky="NWES",
            padx=5,
            pady=5,
        )
        dataframe.columnconfigure(0, weight=1)  # type: ignore
        dataframe.rowconfigure(0, weight=1)  # type: ignore
        ttk.Label(
            dataframe,
            text="Data for {}".format(str(filename)),
            font="helvetica 12 bold",
            anchor="center",
        ).grid(column=0, row=0, sticky="NWE")

        # Read the output data
        self.df: _csv._reader = csv.reader(open(filename), delimiter=",")
        listfile: List[List[str]] = list(self.df)
        data_formatter: List[str] = []
        for item in listfile:
            data_formatter.append(" ".join(map(str, item)))
        data_output: str = "\n".join(map(str, data_formatter))

        # Display it
        self.datatext = tkinter.Text(dataframe, width=30, height=33)
        self.datatext.grid(column=0, row=1, sticky="NWE")
        self.datatext.insert("1.0", data_output)
        ttk.Button(dataframe, text="Close", command=self.dataDialog.destroy).grid(
            column=0, row=4, sticky="S"
        )
        self.dataDialog.mainloop()

    def files_select(self) -> None:
        """Allow user to select a directory where datafiles
        are located."""

        filenamesRaw: List[str] = filedialog.askopenfilenames(
            title="Title", filetypes=[("CSV Files", "*.csv"), ("TXT Files", "*.txt")]
        )  # type: ignore
        self.filenames_ = list(filenamesRaw)
        self.dir_selected.set(1)


class PeakLogicFiles:
    """This is the internal logic of Peak Finder.

    PeaklogicFiles looks at a user-provided list of files, then fits
    the square wave voltammogram data inside to a non-linear polynomial
    plus gaussian function, subtracts the polynomial, and reports the
    peak current.  Ultimately, it prints a pretty graph of the data."""

    def __init__(self, app: PeakFinderApp) -> None:
        """Initialize PeakLogic, passing the gui object to this class."""

        self.app: PeakFinderApp = app

    def peakfinder(self) -> List[float]:
        """PeakLogic.peakfinder is the core function to find and report
        peak current values."""

        # Make sure the user has selected files
        if int(self.app.dir_selected.get()) != 1:
            return []

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
            return iplist

        # Otherwise, show the user what was found
        self.app.output.set("Wrote output to {}.csv".format(filename))
        # mainGraph =
        PointBrowser(self.app, self, printing_list, iplist, filename)
        return iplist

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
        prefix: str, center: float, amplitude: float = 0.005, sigma: float = 0.05
    ) -> Tuple[LorentzianModel, Any]:
        peak = LorentzianModel(prefix=prefix)
        pars = peak.make_params()
        pars[prefix + "center"].set(center)
        pars[prefix + "amplitude"].set(amplitude, min=0)
        pars[prefix + "sigma"].set(sigma, min=0)
        return peak, pars

    def fitting_math(
        self,
        xfile: List[str],
        yfile: List[str],
        flag: int = 1,
    ) -> Any:
        """PeakLogic.fitting_math() fits the data to a cosh and a
        gaussian, then subtracts the cosh to find peak current.."""

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

            rough_peak_positions = [min(passingx), center]

            min_y = float(min(passingy))
            model = LinearModel(prefix="Background")
            params = model.make_params()  # a=0, b=0, c=0
            params.add("slope", 0, min=0)
            # params.add("b", 0, min=0)
            params.add("intercept", 0, min=min_y)

            for i, cen in enumerate(rough_peak_positions):
                peak, pars = self.add_lz_peak(f"Peak_{i+1}", cen)
                model = model + peak
                params.update(pars)

            _ = model.eval(params, x=passingx)
            result = model.fit(passingy, params, x=passingx)
            comps = result.eval_components()

            ip = float(max(comps["Peak_2"]))

            if flag == 1:
                return ip
            if flag == 0:
                return (
                    x,
                    y,
                    result.best_fit,
                    comps["Background"],
                    comps["Peak_1"],
                    comps["Peak_2"],
                    ip,
                    passingx,
                )

        except Exception:  # pragma: no cover
            print("Error Fitting")
            print(sys.exc_info())
            return -1

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
        return px, py  # return partial x and partial y

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
                y_peak1: "np.ndarray[Any, np.dtype[np.float64]]"
                y_peak2: "np.ndarray[Any, np.dtype[np.float64]]"
                ip: float
                px: "np.ndarray[Any, np.dtype[np.float64]]"
                x, y, y_fp, y_bkg, y_peak1, y_peak2, ip, px = self.fitting_math(
                    x_list, y_list, flag=0
                )
                self.test_grapher(x, y, y_fp, y_bkg, y_peak1, y_peak2, file, ip, px)

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
        y_peak1: "np.ndarray[Any, np.dtype[np.float64]]",
        y_peak2: "np.ndarray[Any, np.dtype[np.float64]]",
        file: str,
        ip: float,
        px: "np.ndarray[Any, np.dtype[np.float64]]",
    ) -> None:  # pragma: no cover
        """PeakLogic.test_grapher() displays a graph of the test fitting."""

        plt.close(2)  # close previous test if open

        # add background components
        full_bkg: "np.ndarray[Any, np.dtype[np.float64]]" = y_bkg + y_peak1

        file_name: str = os.path.basename(file)
        self.fig2 = plt.figure(2)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.plot(x, y, "r.", label="data")
        self.ax2.plot(px, y_fp, label="fit")
        self.ax2.plot(px, full_bkg, label="background")
        # self.ax2.plot(px, y_bkg, label="background")
        # self.ax2.plot(px, y_peak1, label="peak 1")
        self.ax2.plot(px, y_peak2, label="methylene blue")
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


class ProgressBar:  # pragma: no cover
    """Create a tkinter Progress bar widget."""

    def __init__(
        self,
        root: ttk.Frame,
        width: float = 100,
        height: float = 10,
        maxval: float = 100,
    ) -> None:
        """Initialize ProgressBar to make the tkinter progressbar."""

        self.root = root
        self.maxval: float = float(maxval)
        self.canvas = tkinter.Canvas(
            self.root,
            width=width,
            height=height,
            highlightthickness=0,
            relief="ridge",
            borderwidth=2,
        )
        self.canvas.create_rectangle(0, 0, 0, 0, fill="blue")
        ttk.Label(self.root, text="Progress:").grid(column=1, row=14, sticky="W")
        self.canvas.grid(column=1, row=15, sticky="WE", columnspan=3)

    def set_maxval(self, maxval: float) -> None:
        """ProgressBar.set_maxval() sets the max value of the
        progressbar."""

        self.maxval = float(maxval)

    def update(self, value: float = 0) -> None:
        """ProgressBar.update() updates the progressbar to a specified
        value."""

        if value < 0:
            value = 0
        elif value > self.maxval:
            value = self.maxval
        self.canvas.delete(tkinter.ALL)
        self.canvas.create_rectangle(
            0,
            0,
            self.canvas.winfo_width() * value / self.maxval,
            self.canvas.winfo_reqheight(),
            fill="blue",
        )
        self.root.update()


def main() -> None:  # pragma: no cover
    """Entry point for gui script."""
    PeakFinderApp()


# Main magic
if __name__ == "__main__":  # pragma: no cover
    main()
