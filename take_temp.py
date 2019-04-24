#!/usr/bin/env python3
import numpy as np
import time
import datetime as datetime
from sense_hat import SenseHat
import subprocess as sp
import sys
import argparse
import temp_vars

sense = SenseHat()
temperature_data = []

# -- parse args --
par = argparse.ArgumentParser(description="temperature study run number")
arg = par.add_argument
arg("-r", "--run", type=str, help="set run number")
args = vars(par.parse_args())

# -- set parameters --
run_num = None

if args["run"]:
    run_num = args["run"]

file_name = 'temperature_data_run%s.txt' % (run_num)

def get_temperatures():
    on_LEDs()
    #for pos in range(1,6):
    for pos in range(1,15):
        curr_temp = float(sense.get_temperature())
        print("Temperature: %f" % (curr_temp))
        curr_time = time.time()
        curr = [curr_time, curr_temp]
        temperature_data.append(curr)
        time.sleep(1)
    sense.clear()
    return temperature_data

def on_LEDs():
    o = (0,0,0)
    x = (0,255,0)
    on_pixel_matrix = [
        o, o, o, o, o, o, o, o,
        o, o, o, o, o, o, o, o,
        x, x, x, o, x, o, o, x,
        x, o, x, o, x, x, o, x,
        x, o, x, o, x, o, x, x,
        x, x, x, o, x, o, o, x,
        o, o, o, o, o, o, o, o,
        o, o, o, o, o, o, o, o,
    ]
    sense.set_pixels(on_pixel_matrix)

def write_temperature(data):
    f = open(file_name,'w')
    for i in range(len(data) - 1):
        str = "%f %f" % (data[i][0], data[i][1])
        f.write(str)
    f.close()

def main():
    write_temperature(get_temperatures())
    sp.Popen(['scp', file_name, 'keirahansen@192.168.0.22:Desktop/research/Code/Temp_study'])
    print('%s synced to Desktop/coherent/Analysis/temp' % (file_name))

def __name__=='__main__':
    main()
