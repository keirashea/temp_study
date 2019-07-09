#!/usr/bin/env python3
import time
from array import array
import numpy as np
from ROOT import TFile, TChain, TTree

class root_data_collector:
    filename = ''
    run_number = 0
    energies = []
    calibrated_energies = []
    times = []
    temperatures = []
    temperature_average = 0
    temperature_var = 0
    temperature_std = 0
    temperature_max = 0
    temperature_min = 0

    def __init__(self, filename):
        self.filename = filename

        def find_branch_data(file, branch):
            """
            General function to retrieve data from a given Branch in given root file.
            """
            tfile = TFile(file)
            ttree = tfile.Get('st')
            total_entries = ttree.GetEntries()
            branch_entries = array('d',[0])
            ttree.SetBranchAddress(branch, branch_entries)
            branch_data = []
            for i in range(total_entries):
                ttree.GetEntry(i)
                branch_data.append(branch_entries[0])
            ttree.ResetBranchAddresses()
            return branch_data

        def set_run_number(file):
            """
            Returns the run number for given root file.
            """
            'runs/root_runs/NaI_ET_run####.root'
            run_number = file[25:29]
            return run_number

        def set_time_data(file):
            """
            Returns a list of timestamps, in seconds, for each energy measurement from a
            given root file.
            """
            start_time_data = find_branch_data(file,'t0')
            start_time = start_time_data[0]
            entry_time_data = find_branch_data(file, 'time')
            time_data = []
            for i in range(len(entry_time_data)):
                entry_time = entry_time_data[i] / 10e+07
                time_data.append(entry_time + start_time)
            return time_data

        def set_energy_data(file):
            """
            Returns a list of energy measurements from a given root file.
            """
            energy_data = find_branch_data(file, 'energy')
            return energy_data

        print('Retrieving run number.')
        self.run_number = set_run_number(filename)
        print('Retrieving energy data.')
        self.energies = set_energy_data(filename)
        print('Retrieving time data.')
        self.times = set_time_data(filename)

    def add_temperatures(self, temps):
        temp_index = 0
        temp_list = []
        print('Adding temperatures to energy entries.')
        while temps.times[temp_index] < self.times[0]:
            temp_index += 1
        for i in range(len(self.energies)):
            if abs(self.times[i] - temps.times[temp_index]) > 30:
                temp_index += 1
            temp_list.append(temps.temperatures[temp_index])
        self.temperatures = temp_list
        temp_array = np.array(self.temperatures)
        self.temperature_average = np.mean(temp_array)
        self.temperature_var = np.var(temp_array)
        self.temperature_std = np.std(temp_array)
        self.temperature_max = temp_array.max()
        self.temperature_min = temp_array.min()

    def calibrate_energies(self, voltage, cal_pars):
        print('Calibrating energy entries.')
        self.calibrated_energies.clear()
        offset = cal_pars[0]
        slope = cal_pars[1]
        saturation = cal_pars[2]
        cal = np.exp(offset + slope * voltage + saturation * (voltage ** 2))
        for uncalibrated_energy in self.energies:
            calibrated_energy = uncalibrated_energy / cal
            self.calibrated_energies.append(calibrated_energy)

class temperature_data_collector:
    run_number = 0
    temperatures = []
    times = []

    def __init__(self, filename):
        'runs/temperature_runs/run####_temperature_data.txt'
        print('Retrieving temperature data.')
        self.run_number = filename[25:29]
        self.filename = filename
        temp_data_file = open(filename, 'r')
        lines = temp_data_file.readlines()
        for line in lines:
            hold = line.split(' ')
            self.times.append(float(hold[0]))
            self.temperatures.append(float(hold[1]))
        temp_data_file.close()
