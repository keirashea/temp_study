#!/usr/bin/env python3
from ROOT import TFile, TChain, TTree
from array import array
import time
import datetime as datetime
import numpy as np
import matplotlib.pyplot as plt
import temp_vars

def main(argv):
    filename='NaI_ET_run2532.root'

    """
    NaI_ET_run####.root
        branch: runNumber       -> run identifier
        branch: subRunNumber    -> always zero
        branch: runningState    -> always running
        branch: energy          -> ADC (uncalibrated energy)
        branch: amplitude       -> amplitude of waveform (max - min)
        branch: time            -> time increment of 10 min run
        branch: t0              -> unix time at start of run
        branch: ChannelNumber   -> digitizer channel number
        branch: peakingTime     -> set value in ORCA
    """

    # runList = []
    # file_dir = "./data"
    # tchain_ref = TChain("st")
    # for run in runList:
    #     f_name = "%s/run_%d/FILTERED/compassF_run_%d.root" % (file_dir, run, run)
    #     runtime += get_runtime(f_name)
    #     ch.Add(f_name)

    tfile = TFile(file_name)
    ttree = tfile.Get("st")
    total_entries = ttree.GetEntries()

    start_time_entries = array('l',[0])
    ttree.SetBranchAddress("t0", start_time_entries)
    ttree.GetEntry(0)
    start_time = start_time_entries[0]

    energy_data = get_energy_data(ttree, total_entries)
    temp_data = get_temp_data()
    plot_gainvtemp()

def get_energy_data(ttree_ref, n_entries):
    """
    Returns a list of uncalibrated (ADC) energies and times from root file
    in the form [time,energy]
    """
    energy_data_ref = np.empty([n_entries,2], dtype=float)
    # create a pointer to an event object. This will be used
    # to read the branch values.
    entry_pointer = array('l', [0])

    # get two branches and set the branch address
    energy_branch = ttree_ref.GetBranch("energy")
    time_branch = ttree_ref.GetBranch("time")
    energy_branch.SetAddress(entry_pointer)
    time_branch.SetAddress(entry_pointer)

    # populate energy_data with entry from branches
    for i in range(total_entries):
        energy_data_ref[i] = [time_branch.GetEntry(i),energy_branch.GetEntry(i)]

    return energy_data_ref

def get_temp_data():
    """
    Returns list of temperatures and times from temperature.txt files
    from RPi in the format [time,temp]
    """
    temp_data_file = open('temperature_data.txt', 'r')
    lines = temp_data_file.readlines()
    temp_data_ref = []
    for line in lines:
        hold = line.split(" ")
        temp_data_ref.append([float(hold[0]), float(hold[1])])
    temp_data_file.close()
    return temp_data_ref

#def calibrate_energy():
    """
    Calibrate ADC energy to 208Tl and 40K
    """

#def find_gain():
    """
    Calculate gain (gain_offset from Calibration.cc)
    """

#def find_resolution():
    """
    Calculate resolution of 40K
    """

#def plot_gainVtemp():
    """
    Plots uncalibrated energy on y-axis and time on x-axis, and plots
    """
    plt.cla() # clear plot from last time
    plt.plot(energy_data[0:], energy_data[1:], 'k.', temp_data[0:], energy_data[1:], 'r.')
    plt.xlabel("Time")
    #plt.ylabel("Counts")
    plt.show()
    #plt.savefig("...")
    plt.clf()

#def plot_resVtemp():
    """
    Plots resolution of 40K peak
    """

if __name__=='__main__':
    main()
