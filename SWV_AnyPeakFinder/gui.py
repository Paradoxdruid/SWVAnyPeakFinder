#!/usr/bin/env python3

"""
SWV Any PeakFinder.

This program finds peak height values (i.e., peak currents) from .txt files
and .csv files containing squarewave voltammogram data, using any
selected files.
"""


from __future__ import absolute_import

import csv
import os
import platform
import tkinter
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any, List, Tuple

import _csv
import matplotlib.pyplot as plt

from SWV_AnyPeakFinder.__version__ import __author__, __copyright__, __version__
from SWV_AnyPeakFinder.logic import PeakLogicFiles

if platform.system() == "Darwin":  # pragma: no cover
    import matplotlib

    matplotlib.use("TkAgg")


class PeakFinderApp(tkinter.Tk):  # pragma: no cover
    """This is the gui for Peak Finder.

    This application displays a minimal user interface to select a
    directory and to specify file formats and output filename, then
    reports on the programs function with a simple progressbar."""

    def __init__(self) -> None:
        """set up the peak finder app GUI."""

        self.directory_manager()
        tkinter.Tk.__init__(self)
        self.window_title: str = "SWV AnyPeakFinder {}".format(str(__version__))

        # invoke PeakLogicFiles to do the actual work
        self.logic: PeakLogicFiles = PeakLogicFiles(self)

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
        self.peak_center_: tkinter.DoubleVar = tkinter.DoubleVar(value=-0.3)
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
        ttk.Button(mainframe, text="Find Peaks", command=self.run_peakfinder).grid(
            column=1, row=10, sticky="W"
        )

        # Display test fit button
        ttk.Button(mainframe, text="Test fit", command=self.logic.test_fit).grid(
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
            ).format(str(__author__), str(__copyright__)),
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
        dataframe = ttk.Frame(self.dataDialog, width="600", height="600")
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

    def run_peakfinder(self) -> None:
        if peak_output := self.logic.peakfinder():
            printing_list, iplist, filename = peak_output
            self.output.set("Wrote output to {}.csv".format(filename))
            PointBrowser(self, self.logic, printing_list, iplist, filename)
        else:
            self.output.set("Failure")


# ##############################################################################


class PointBrowser:  # pragma: no cover
    """This is the class that draws the main graph of data for Peak Finder.

    This class creates a line and point graph with clickable points.
    When a point is clicked, it calls the normal test Fit display
    from the main logic."""

    def __init__(
        self,
        app: PeakFinderApp,
        logic: PeakLogicFiles,
        xticksRaw: List[int],
        y: List[float],
        fileTitle: str,
    ) -> None:
        """Create the main output graph and make it clickable."""

        self.app: PeakFinderApp = app
        self.logic: PeakLogicFiles = logic
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
        self.button.on_clicked(self.app.data_popup)

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


# ##############################################################################


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
