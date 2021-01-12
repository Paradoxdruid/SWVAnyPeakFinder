#!/usr/bin/env python3

"""This program finds peak height values (i.e., peak currents) from .txt files
and .csv files containing squarewave voltammogram data, using any
selected files. """

__author__ = "Andrew J. Bonham"
__copyright__ = "Copyright 2010-2019, Andrew J. Bonham"
__credits__ = ["Andrew J. Bonham"]
__version__ = "1.6.6"
__maintainer__ = "Andrew J. Bonham"
__email__ = "bonham@gmail.com"
__status__ = "Production"
__description__ = (
    "GUI application for resolving peak heights in square-wave voltammetry datafiles."
)
__license__ = "GPLv3"
__url__ = "https://github.com/Paradoxdruid/SWVAnyPeakFinder"

# Setup: import basic modules we need

import csv
import os
import platform
import re
import sys
import numpy
from scipy.optimize import least_squares
import pylab
import tkinter
from tkinter import filedialog as tkFileDialog
from tkinter import ttk
from typing import List, Tuple, Any

if platform.system() == "Darwin":
    import matplotlib

    matplotlib.use("TkAgg")

# Define our Classes


class PointBrowser:

    """This is the class that draws the main graph of data for Peak Finder.

    This class creates a line and point graph with clickable points.
    When a point is clicked, it calls the normal test Fit display
    from the main logic."""

    def __init__(
        self,
        app: "PeakFinderApp",
        logic: "PeakLogicFiles",
        xticksRaw: numpy.ndarray,
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
        self.fig = pylab.figure(1)
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(self.x, self.y, "bo-", picker=5)
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
        self.axb = pylab.axes([0.75, 0.03, 0.15, 0.05])
        self.button = pylab.Button(self.axb, "Results")
        # self.ax.plot._test = self.button
        # self.button.on_clicked(self.app.data_popup)

        # Draw the figure
        self.fig.canvas.draw()
        pylab.show()

    def onpick(self, event: Any) -> None:
        """Capture the click event, find the corresponding data
        point, then update accordingly."""

        thisone = event.artist
        x = thisone.get_xdata()
        ind = event.ind

        dataind = x[ind[0]]

        self.logic.test_fit(dataind)

        self.fig.canvas.draw()
        pylab.show()

    def update(self) -> None:
        """Update the main graph and call my response function."""

        if self.lastind is None:
            return
        dataind: int = self.lastind

        self.selected.set_visible(True)
        self.selected.set_data(self.x[dataind], self.y[dataind])

        self.logic.test_fit(dataind)

        self.fig.canvas.draw()
        pylab.show()


class PeakFinderApp(tkinter.Tk):

    """This is the gui for Peak Finder.

    This application displays a minimal user interface to select a
    directory and to specify file formats and output filename, then
    reports on the programs function with a simple progressbar."""

    def __init__(self) -> None:
        """set up the peak finder app GUI."""

        self.directory_manager()
        tkinter.Tk.__init__(self)
        self.window_title = "SWV AnyPeakFinder {}".format(str(__version__))

        # invoke PeakLogicFiles to do the actual work
        logic: "PeakLogicFiles" = PeakLogicFiles(self)

        # create the frame for the main window
        self.title(self.window_title)
        self.geometry("+100+100")
        self.resizable(tkinter.FALSE, tkinter.FALSE)
        mainframe = ttk.Frame(self)
        mainframe.grid(
            column=0,
            row=0,
            sticky=(tkinter.N, tkinter.W, tkinter.E, tkinter.S),
            padx=5,
            pady=5,
        )
        mainframe.columnconfigure(0, weight=1)
        mainframe.rowconfigure(0, weight=1)

        # define our variables in tkinter-form and give sane defaults
        self.filename_ = tkinter.StringVar(value="output1")
        self.output = tkinter.StringVar()
        self.filenames_: List[str] = []
        self.dir_selected = tkinter.IntVar(value=0)
        self.init_potential_ = tkinter.DoubleVar(value=-0.2)
        self.final_potential_ = tkinter.DoubleVar(value=-0.4)
        self.peak_center_ = tkinter.DoubleVar(value=-0.3)
        self.final_edge_ = tkinter.DoubleVar(value=-1)
        self.init_edge_ = tkinter.DoubleVar(value=1)
        self.guesses_ = None

        # display the entry boxes
        ttk.Label(mainframe, text=self.window_title, font="helvetica 12 bold").grid(
            column=1, row=1, sticky=tkinter.W, columnspan=2
        )

        self.filename_entry = ttk.Entry(
            mainframe, width=12, textvariable=self.filename_
        ).grid(column=2, row=2, sticky=(tkinter.W, tkinter.E))
        ttk.Label(mainframe, text="Filename:").grid(column=1, row=2, sticky=tkinter.W)

        self.init_potential = ttk.Entry(
            mainframe, width=12, textvariable=self.init_potential_
        ).grid(column=2, row=7, sticky=(tkinter.W, tkinter.E))
        ttk.Label(mainframe, text="Initial Potential:").grid(
            column=1, row=7, sticky=tkinter.W
        )

        self.final_potential = ttk.Entry(
            mainframe, width=12, textvariable=self.final_potential_
        ).grid(column=2, row=8, sticky=(tkinter.W, tkinter.E))
        ttk.Label(mainframe, text="Final Potential:").grid(
            column=1, row=8, sticky=tkinter.W
        )

        self.peak_center = ttk.Entry(
            mainframe, width=12, textvariable=self.peak_center_
        ).grid(column=2, row=9, sticky=(tkinter.W, tkinter.E))
        ttk.Label(mainframe, text="Peak Center:").grid(
            column=1, row=9, sticky=tkinter.W
        )

        ttk.Label(mainframe, text="Peak Location Guess", font="helvetica 10 bold").grid(
            column=1, row=5, sticky=tkinter.W
        )
        ttk.Label(mainframe, text="(only change if necessary)").grid(
            column=1, row=6, sticky=tkinter.W
        )

        # Display Generate button
        ttk.Button(mainframe, text="Find Peaks", command=logic.peakfinder).grid(
            column=1, row=10, sticky=tkinter.W
        )

        # Display test fit button
        ttk.Button(mainframe, text="Test fit", command=logic.test_fit).grid(
            column=2, row=10, sticky=tkinter.W
        )

        # Display Directory button
        ttk.Button(mainframe, text="Select Files", command=self.files_select).grid(
            column=1, row=4, sticky=tkinter.W
        )

        # Show the output
        ttk.Label(mainframe, textvariable=self.output).grid(
            column=1, row=13, sticky=(tkinter.W, tkinter.E), columnspan=5
        )

        # Set up Outlier correction
        ttk.Label(mainframe, text="Regions to include (from -> to):").grid(
            column=1, row=12, sticky=tkinter.W
        )
        self.final_edge = ttk.Entry(
            mainframe, width=12, textvariable=self.final_edge_
        ).grid(column=1, row=13, sticky=(tkinter.E))
        self.init_edge = ttk.Entry(
            mainframe, width=12, textvariable=self.init_edge_
        ).grid(column=2, row=13, sticky=(tkinter.W, tkinter.E))

        # Pad the windows for prettiness
        for child in mainframe.winfo_children():
            child.grid_configure(padx=5, pady=5)

        # Display a progressbar
        self.bar: "ProgressBar" = ProgressBar(mainframe, width=100, height=10)
        self.bar.update(0)

        ttk.Label(mainframe, text="").grid(
            column=1, row=16, sticky=(tkinter.W, tkinter.E)
        )

        # add a system menu
        self.option_add("*tearOff", tkinter.FALSE)

        self.menubar = tkinter.Menu(self)
        self["menu"] = self.menubar

        self.menu_file = tkinter.Menu(self.menubar)
        self.menubar.add_cascade(menu=self.menu_file, label="File")

        self.menu_file.add_command(label="Exit", command=self.destroy)

        self.menu_help = tkinter.Menu(self.menubar, name="help")
        self.menubar.add_cascade(menu=self.menu_help, label="Help")

        self.menu_help.add_command(label="How to Use", command=self.help_popup)
        self.menu_help.add_command(
            label="About Peak Finder...", command=self.about_popup
        )

        # Run the program
        pylab.show()
        self.mainloop()

    def directory_manager(self) -> None:
        """Set the initial directory to the users home directory."""

        if platform.system() == "Windows":
            # import windows file management
            from pathlib import Path

            self.mydocs: Any = str(Path.home())
        else:
            # import mac/linux file management
            self.mydocs: Any = os.getenv("HOME")
        os.chdir(self.mydocs)

    def about_popup(self) -> None:
        """Display a pop-up menu about the program."""

        self.aboutDialog = tkinter.Toplevel(self)
        self.aboutDialog.resizable(tkinter.FALSE, tkinter.FALSE)
        self.aboutDialog.geometry("+400+100")

        aboutframe = ttk.Frame(self.aboutDialog, width="200", height="200")
        aboutframe.grid_propagate(0)
        aboutframe.grid(
            column=0,
            row=0,
            sticky=(tkinter.N, tkinter.W, tkinter.E, tkinter.S),
            padx=5,
            pady=5,
        )
        aboutframe.columnconfigure(0, weight=1)
        aboutframe.rowconfigure(0, weight=1)
        ttk.Label(
            aboutframe,
            text=self.window_title,
            font="helvetica 12 bold",
            anchor="center",
        ).grid(column=0, row=0, sticky=(tkinter.N, tkinter.W, tkinter.E))
        ttk.Label(
            aboutframe,
            text=(
                "Voltammogram data analysis\nsoftware for "
                "CH Instruments data.\n\nWritten by\n{0}\n"
                "http://www.andrewjbonham.com\n{1}\n\n\n"
            ).format(str(__author__), str(__copyright__)),
            anchor="center",
            justify="center",
        ).grid(column=0, row=1, sticky=tkinter.N)
        ttk.Button(aboutframe, text="Close", command=self.aboutDialog.destroy).grid(
            column=0, row=4, sticky=(tkinter.S)
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
            sticky=(tkinter.N, tkinter.W, tkinter.E, tkinter.S),
            padx=5,
            pady=5,
        )
        helpframe.columnconfigure(0, weight=1)
        helpframe.rowconfigure(0, weight=1)
        ttk.Label(
            helpframe,
            text=("{} Help".format(self.window_title)),
            font="helvetica 12 bold",
            anchor="center",
        ).grid(column=0, row=0, sticky=(tkinter.N, tkinter.W, tkinter.E))
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
        dataframe.grid_propagate(0)
        dataframe.grid(
            column=0,
            row=0,
            sticky=(tkinter.N, tkinter.W, tkinter.E, tkinter.S),
            padx=5,
            pady=5,
        )
        dataframe.columnconfigure(0, weight=1)
        dataframe.rowconfigure(0, weight=1)
        ttk.Label(
            dataframe,
            text="Data for {}".format(str(filename)),
            font="helvetica 12 bold",
            anchor="center",
        ).grid(column=0, row=0, sticky=(tkinter.N, tkinter.W, tkinter.E))

        # Read the output data
        self.df = csv.reader(open(filename), delimiter=",")
        listfile: List[List[str]] = list(self.df)
        data_formatter: List[str] = []
        for item in listfile:
            data_formatter.append(" ".join(map(str, item)))
        data_output: str = "\n".join(map(str, data_formatter))

        # Display it
        self.datatext = tkinter.Text(dataframe, width=30, height=33)
        self.datatext.grid(column=0, row=1, sticky=(tkinter.N, tkinter.W, tkinter.E))
        self.datatext.insert("1.0", data_output)
        ttk.Button(dataframe, text="Close", command=self.dataDialog.destroy).grid(
            column=0, row=4, sticky=(tkinter.S)
        )
        self.dataDialog.mainloop()

    def files_select(self) -> None:
        """Allow user to select a directory where datafiles
        are located."""

        # filenamesRaw = tkinter.filedialog.askopenfilenames(
        filenamesRaw = tkFileDialog.askopenfilenames(
            title="Title", filetypes=[("CSV Files", "*.csv"), ("TXT Files", "*.txt")]
        )
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
            self.g.write(
                "Init Potential,{}\n".format(str(self.app.init_potential_.get()))
            )
            self.g.write(
                "Final Potential,{}\n".format(str(self.app.final_potential_.get()))
            )
            self.g.write("Peak Center,{}\n".format(str(self.app.peak_center_.get())))
            self.g.write("Left Edge,{}\n".format(str(self.app.init_edge_.get())))
            self.g.write("Right Edge,{}\n\n".format(str(self.app.final_edge_.get())))
            self.g.write("--------,--------\n")
            self.g.write("time,file,peak current\n")

            # run the peakfinder
            printing_list: List
            iplist: List
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

        pylab.close(2)  # close test fitting graph if open

        for each in filenamesList:  # loop through each file
            try:
                dialect = csv.Sniffer().sniff(open(each).read(1024), delimiters="\t,")
                open(each).seek(0)
                self.f = csv.reader(open(each), dialect)
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

    def peak_math(self, listsx: numpy.ndarray, listsy: numpy.ndarray) -> List[float]:
        """PeakLogic.peak_math() passes each data file to .fitting_math,
        and returns a list of peak currents."""

        iplist: List[float] = []
        count: int = 1
        # give reasonable starting values for non-linear regression
        if self.app.guesses_ is None:
            starting_v0: List[float] = [1e-6, 1e-6, 1e-7, 1e-7, -0.3, 1e-8]
        else:
            starting_v0: List[float] = self.app.guesses_

        for xfile, yfile in zip(listsx, listsy):
            if count == 1:
                ip: float
                v0: List[float]
                ip, v0 = self.fitting_math(xfile, yfile, starting_v0, 1)
            else:
                ip, _v0 = self.fitting_math(xfile, yfile, v0, 1)

            # check data quality
            if ip < 0:
                ip = 0
            iplist.append(ip)
            self.app.bar.update(count)
            count = count + 1

        return iplist

    def fitting_math(
        self,
        xfile: numpy.ndarray,
        yfile: numpy.ndarray,
        guesses: List[float],
        flag: int = 1,
    ) -> Any:
        """PeakLogic.fitting_math() fits the data to a cosh and a
        gaussian, then subtracts the cosh to find peak current.."""

        try:
            init_pot: float = self.app.init_potential_.get()
            final_pot: float = self.app.final_potential_.get()
            edgelength: float = numpy.abs(init_pot - final_pot)
            center: float = self.app.peak_center_.get()
            x: numpy.ndarray = numpy.array(xfile, dtype=numpy.float64)
            y: numpy.ndarray = numpy.array(yfile, dtype=numpy.float64)

            # fp is full portion with the exp / cosh plus gaussian
            def fp(v: List[float], x: numpy.ndarray) -> numpy.ndarray:
                return (
                    (v[0] * (x ** 2))
                    + (v[1] * x)
                    + v[2]
                    + v[3] * numpy.exp(-(((x - v[4]) ** 2) / (2 * v[5] ** 2)))
                )

            # pp is just the exp / cosh portion
            def pp(v: List[float], x: numpy.ndarray) -> numpy.ndarray:
                return (v[0] * (x ** 2)) + (v[1] * x) + v[2]

            # e is the error of the full fit from the real data
            def e(v: List[float], x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
                return fp(v, x) - y

            # cut out outliers
            passingx: numpy.ndarray
            passingy: numpy.ndarray
            passingx, passingy = self.trunc_edges(xfile, yfile)

            # cut out the middle values and return the edges
            _outx: numpy.ndarray
            outy: numpy.ndarray
            _outx, outy = self.trunc_list(passingx, passingy)
            AA: float = numpy.average(outy)
            less: numpy.ndarray = passingx < init_pot
            greater: numpy.ndarray = passingx > final_pot
            PeakHeight: float
            try:
                PeakHeight = numpy.max(passingy[less & greater])
            except ValueError:
                PeakHeight = numpy.max(passingy)

            # check if guesses
            if guesses is None:
                guesses = [1e-6, 1e-6, PeakHeight, AA, center, edgelength / 6]

            # fit the background and baseline to all data
            _results: Any = least_squares(
                e,
                guesses,
                args=(passingx, passingy),
                bounds=(  # constain an upward facing parabola
                    [0, -numpy.inf, -numpy.inf, -numpy.inf, -numpy.inf, -numpy.inf],
                    numpy.inf,
                ),
            )
            v: Any = _results.x

            ip: numpy.ndarray = fp(v, v[4]) - pp(v, v[4])

            if flag == 1:
                return ip, v
            if flag == 0:
                return x, y, fp(v, passingx), pp(v, passingx), ip, passingx, v

        except Exception:
            print("Error Fitting")
            print(sys.exc_info())
            return -1

    def trunc_list(
        self, listx: numpy.ndarray, listy: numpy.ndarray
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Remove the central portions of an x-y data list (where we
        suspect the gaussian is found)."""

        newx: List[float] = []
        newy: List[float] = []
        start_spot: str
        start_h: str
        for start_spot, start_h in zip(listx, listy):
            spot: float = float(start_spot)
            h: float = float(start_h)
            low: float = float(self.app.final_potential_.get())
            high: float = float(self.app.init_potential_.get())
            if spot < low:  # add low values
                newx.append(spot)
                newy.append(h)
            else:
                pass
            if spot > high:  # add high values
                newx.append(spot)
                newy.append(h)
            else:
                pass

        px: numpy.ndarray = numpy.array(newx, dtype=numpy.float64)
        py: numpy.ndarray = numpy.array(newy, dtype=numpy.float64)
        return px, py

    def trunc_edges(
        self, listx: numpy.ndarray, listy: numpy.ndarray
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
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
                else:
                    pass
            else:
                pass

        # convert results back to an array
        px: numpy.ndarray = numpy.array(newx, dtype=numpy.float64)
        py: numpy.ndarray = numpy.array(newy, dtype=numpy.float64)
        return px, py  # return partial x and partial y

    def test_fit(self, dataind: int = 0) -> None:
        """Perform a fit for the first data point and display it for
        the user."""

        # Make sure the user has selected a directory
        if int(self.app.dir_selected.get()) == 1:
            try:
                filenamesList: List[str] = self.app.filenames_
                file: str = filenamesList[dataind]
                dialect: Any = csv.Sniffer().sniff(
                    open(file).read(1024), delimiters="\t,"
                )

                open(file).seek(0)  # open the first data file
                self.testfile: Any = csv.reader(open(file), dialect)
                listfile: List[str] = list(self.testfile)

                # remove the header rows from the file, leaving just the data
                start_pattern: int = 3
                for index, line in enumerate(listfile):
                    try:
                        if line[0]:
                            if re.match("Potential*", line[0]):
                                start_pattern = index
                    except IndexError:
                        pass
                datalist: List[str] = listfile[start_pattern + 2 :]

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

                x: numpy.ndarray
                y: numpy.ndarray
                y_fp: numpy.ndarray
                y_pp: numpy.ndarray
                ip: float
                px: numpy.ndarray
                x, y, y_fp, y_pp, ip, px, v = self.fitting_math(
                    x_list, y_list, guesses=None, flag=0
                )
                self.app.guesses_ = v
                self.test_grapher(x, y, y_fp, y_pp, file, ip, px)

            except (ValueError, IndexError):
                pass
        else:
            pass

    def test_grapher(
        self,
        x: numpy.ndarray,
        y: numpy.ndarray,
        y_fp: numpy.ndarray,
        y_pp: numpy.ndarray,
        file: str,
        ip: float,
        px: numpy.ndarray,
    ) -> None:
        """PeakLogic.test_grapher() displays a graph of the test fitting."""

        pylab.close(2)  # close previous test if open

        file_name: str = os.path.basename(file)
        self.fig2 = pylab.figure(2)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.plot(x, y, "ro", label="data")
        self.ax2.plot(px, y_fp, label="fit")
        self.ax2.plot(px, y_pp, label="baseline")
        self.ax2.set_xlabel("Potential (V)")
        self.ax2.set_ylabel("Current (A)")
        self.ax2.set_title("Fit of {}".format(str(file_name)))
        self.ax2.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")
        self.ax2.legend()
        self.fig2.subplots_adjust(bottom=0.15)
        self.fig2.subplots_adjust(left=0.15)
        self.text = self.ax2.text(
            0.05,
            0.95,
            "Peak Current:\n%.2e A" % ip,
            transform=self.ax2.transAxes,
            va="top",
        )

        self.fig2.canvas.draw()
        pylab.show()


class ProgressBar:

    """Create a tkinter Progress bar widget."""

    def __init__(
        self, root: Any, width: float = 100, height: float = 10, maxval: float = 100
    ) -> None:
        """Initialize ProgressBar to make the tkinter progressbar."""

        self.root = root
        self.maxval: float = float(maxval)
        self.canvas = tkinter.Canvas(
            self.root,
            width=width,
            height=height,
            highlightt=0,
            relief="ridge",
            borderwidth=2,
        )
        self.canvas.create_rectangle(0, 0, 0, 0, fill="blue")
        self.label = ttk.Label(self.root, text="Progress:").grid(
            column=1, row=14, sticky=tkinter.W
        )
        self.canvas.grid(column=1, row=15, sticky=(tkinter.W, tkinter.E), columnspan=3)

    def set_maxval(self, maxval: float) -> None:
        """ProgressBar.set_maxval() sets the max value of the
        progressbar."""

        self.maxval: float = float(maxval)

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


def main():
    """Entry point for gui script."""
    app: PeakFinderApp = PeakFinderApp()  # noqa


# Main magic
if __name__ == "__main__":
    main()
