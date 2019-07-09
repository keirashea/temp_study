#!/usr/bin/env python3
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ROOT import TH1D, TF1, TCanvas, THStack, TLine
from ROOT import TColor, TStyle, gROOT, gStyle
import data_handling as datastruct
import parameter_handling as parameters

"""
TO DO:
    ☐ add error bars
    ☐ find multiplication factor to scale gains
"""
def main():

    group_2939 = ['2939', '2940', '2941', '2942', '2943', '2944', '2945']
    group_2946 = ['2946', '2947', '2948', '2949', '2950', '2951']
    group_2952 = ['2952', '2953', '2954', '2955', '2956'] #, '2957', '2958', '2959', '2960', '2961']
    group_3238 = ['3238', '3239', '3240', '3241', '3242'] #, '3243', '3244', '3245', '3246', '3247']
    calibration_group = ['2559', '2952', '3238']

    group = group_3238
    group_name = group[0]
    print(len(group), ' files in data set.')

    path = 'output/group_%s' % group_name
    if not os.path.exists(path):
        os.mkdir(path)
    temp_filename = 'runs/temperature_runs/run%s_temperature_data.txt' % (group_name)
    temp_data = datastruct.temperature_data_collector(temp_filename)

    group_data = []
    for run in group:
        run_filename = 'runs/root_runs/NaI_ET_run' + run + '.root'
        print('Collecting data for ', run_filename)
        root_data = datastruct.root_data_collector(run_filename)
        root_data.add_temperatures(temp_data)
        group_data.append(root_data)

    plot_hist(group_data)
    #plot_temp_data(temp_data)
    #plot_correlations(group)
    #plot_calibrations(group_data, 800, calibration_group, stacked=True)
    #plot_gain_curves(group_data, calibration_group)


def make_hist(title, x_data, min, max):
    """
    Returns a root histogram containing energy data from giving root file.
    """
    hist = TH1D('h1d', title, 600, min, max)
    print('Creating histogram for %s.' % (title))
    for i in range(len(x_data)):
        hist.Fill(x_data[i])
    hist.SetTitle(title)
    hist.GetYaxis().SetTitle('Count')
    hist.GetXaxis().SetTitle('Uncalibrated Energy')
    return hist


def find_peak_fit(peak, par0, par1, par2, par3, par4, low, high):
    """
    [0]*exp(-0.5*((x-[1])/[2])^2) + exp([3] + [4]*x)
    """
    print('Finding %s peak fit.' % (peak))
    peak_fit = TF1('%s_fit' % (peak), 'gaus(0)+expo(3)', low, high)
    peak_fit.SetParameters(par0, par1, par2, par3, par4)
    return peak_fit


def write_run_output(group_name, run, K_fit, Tl_fit):
    print('Writing output to file.')
    filename = 'output/group_' + group_name + '/' + str(run.run_number) + '_output.txt'
    outfile = open(filename, 'w')
    outfile.write('Run:\t\t\t' + str(run.run_number) + '\n')
    outfile.write('Start time:\t\t' + time.ctime(run.times[0]) + '\t' + str(run.times[0]) + '\n')
    outfile.write('Start temperature:\t' + str(run.temperatures[0]) + '\n')
    outfile.write('\n')
    outfile.write('Stop time:\t\t' + time.ctime(run.times[-1]) + '\t' + str(run.times[-1]) + '\n')
    outfile.write('Stop temperature:\t' + str(run.temperatures[-1]) + '\n')
    outfile.write('\n')
    outfile.write('Highest temperature:\t' + str(run.temperature_max) + '\n')
    outfile.write('Lowest temperature:\t' + str(run.temperature_min) + '\n')
    outfile.write('Average:\t\t' + str(run.temperature_average) + '\n')
    outfile.write('Variance:\t\t' + str(run.temperature_var) + '\n')
    outfile.write('Standard Deviation:\t' + str(run.temperature_std) + '\n')
    outfile.write('\n')
    outfile.write('40K peak\n')
    outfile.write('Counts:\t\t\t' + str(K_fit.GetParameter(0)) + '\n')
    outfile.write('Counts error:\t\t' + str(K_fit.GetError(0)) + '\n')
    outfile.write('Mu:\t\t\t' + str(K_fit.GetParameter(1)) + '\n')
    outfile.write('Mu error:\t\t' + str(K_fit.GetError(1)) + '\n')
    outfile.write('Sigma:\t\t\t' + str(K_fit.GetParameter(2)) + '\n')
    outfile.write('Sigma error:\t\t' + str(K_fit.GetError(2)) + '\n')
    outfile.write('Intercept:\t\t' + str(K_fit.GetParameter(3)) + '\n')
    outfile.write('Intercept error:\t' + str(K_fit.GetError(3)) + '\n')
    outfile.write('Slope:\t\t\t' + str(K_fit.GetParameter(4)) + '\n')
    outfile.write('Slope error:\t\t' + str(K_fit.GetError(4)) + '\n')
    outfile.write('\n')
    outfile.write('208Tl peak\n')
    outfile.write('Counts:\t\t\t' + str(Tl_fit.GetParameter(0)) + '\n')
    outfile.write('Counts error:\t\t' + str(Tl_fit.GetError(0)) + '\n')
    outfile.write('Mu:\t\t\t' + str(Tl_fit.GetParameter(1)) + '\n')
    outfile.write('Mu error:\t\t' + str(Tl_fit.GetError(1)) + '\n')
    outfile.write('Sigma:\t\t\t' + str(Tl_fit.GetParameter(2)) + '\n')
    outfile.write('Sigma error:\t\t' + str(Tl_fit.GetError(2)) + '\n')
    outfile.write('Intercept:\t\t' + str(Tl_fit.GetParameter(3)) + '\n')
    outfile.write('Intercept error:\t' + str(Tl_fit.GetError(3)) + '\n')
    outfile.write('Slope:\t\t\t' + str(Tl_fit.GetParameter(4)) + '\n')
    outfile.write('Slope error:\t\t' + str(Tl_fit.GetError(4)) + '\n')
    outfile.close()
    print('Output printed to ' + filename)


def plot_hist(group_data, stacked=False, fitted=False):
    group_name = str(group_data[0].run_number)
    group_size = len(group_data)

    if stacked:
        print('Plotting stacked histogram.')
        run_canvas = TCanvas()
        run_canvas.SetLogy()
        gStyle.SetPalette(87)
        if group_name == '2946':
            TColor.InvertPalette()
        hs = THStack('hs', 'Group %s Histograms' % group_name)
        for run in group_data:
            tab = '        '
            title = str(run.run_number) + tab + 'Average temperature: ' + str(round(run.temperature_average, 2))
            hist = make_hist(title, run.energies, 0, 270000)
            hs.Add(hist)
        hs.Draw('nostack PLC')
        run_canvas.BuildLegend()
        path = 'output/group_' + str(group_data[0].run_number) + '/'
        run_canvas.Print((path + str(group_data[0].run_number) + '_stacked_histogram.pdf['))
        run_canvas.Print((path + str(group_data[0].run_number) + '_stacked_histogram.pdf'))
        run_canvas.Print((path + str(group_data[0].run_number) + '_stacked_histogram.pdf]'))

    else:
        hists_and_fits = []
        hist_range = [20000, 60000, 150000, 350000, 550000]
        for i in range(len(group_data)):
            run = group_data[i]
            hist = make_hist(str(run.run_number), run.energies, 0, hist_range[i])
            if fitted:
                K_key = 'K_' + str(run.run_number)
                K_pars = parameters.fit_pars[K_key]
                K_fit = find_peak_fit('K', K_pars[0], K_pars[1], K_pars[2], K_pars[3], K_pars[4],
                                      K_pars[5], K_pars[6])
                Tl_key = 'Tl_' + str(run.run_number)
                Tl_pars = parameters.fit_pars[Tl_key]
                Tl_fit = find_peak_fit('Tl', Tl_pars[0], Tl_pars[1], Tl_pars[2], Tl_pars[3],
                                       Tl_pars[4], Tl_pars[5], Tl_pars[6])
                hists_and_fits.append([hist, K_fit, Tl_fit])
                write_run_output(group_name, run, K_fit, Tl_fit)
            else:
                hists_and_fits.append([hist])

        run_canvas = TCanvas("run_canvas", "run canvas")
        run_canvas.Divide(3, int(group_size / 2))
        file_name = ''
        canvas_index = 1
        for entry in hists_and_fits:
            pad = run_canvas.cd(canvas_index)
            pad.SetLogy()
            run = entry[0]
            if len(entry) == 3:
                print('Plotting fitted histogram.')
                file_name = '_fitted_histograms.pdf'
                fit1 = entry[1]
                fit2 = entry[2]
                run.Fit(fit1, 'LR')
                run.Fit(fit2, 'LR+')
            else:
                print('Plotting histogram.')
                file_name = '_histograms.pdf'
                run.Draw()
            canvas_index += 1

            path = 'output/group_' + group_name + '/'
            run_canvas.Print((path + group_name + file_name + '['))
            run_canvas.Print((path + group_name + file_name))
            run_canvas.Print((path + group_name + file_name + ']'))


def plot_calibrations(group_data, voltage, cal_pars_list, stacked=False):
    group_name = str(group_data[0].run_number)
    run = group_data[2]
    name = str(run.run_number)
    hists_and_lines = []
    for cal_pars in cal_pars_list:
        pars = parameters.gain_pars[cal_pars]
        run.calibrate_energies(voltage, pars)
        hist = make_hist(name, run.calibrated_energies, 0, 3500)
        K_line = TLine(1460.820, 0, 1460.820, 1000000)
        K_line.SetLineColor(632)
        Tl_line = TLine(2614.511, 0, 2614.511, 1000000)
        Tl_line.SetLineColor(632)
        hists_and_lines.append([hist, K_line, Tl_line])

    if stacked:
        print('Plotting stacked calibrated histograms.')
        run_canvas = TCanvas()
        run_canvas.SetLogy()
        colors = [808, 397, 436, 424, 628, 852, 800, 863, 403, 797]
        color_index = 0
        hs = THStack('hs', 'Group %s Histograms' % group_name)
        for entry in hists_and_lines:
            title = str(run.run_number)
            entry[0].SetLineColor(colors[color_index])
            color_index += 1
            hs.Add(entry[0])
        hs.Draw('nostack')
        K_line = TLine(1460.820, 0, 1460.820, 1000000)
        K_line.SetLineColor(632)
        K_line.Draw()
        Tl_line = TLine(2614.511, 0, 2614.511, 1000000)
        Tl_line.SetLineColor(632)
        Tl_line.Draw()
        run_canvas.BuildLegend()
        path = 'output/group_' + str(group_data[0].run_number) + '/'
        run_canvas.Print((path + str(group_data[0].run_number) + '_stacked_calibrated_histogram.pdf['))
        run_canvas.Print((path + str(group_data[0].run_number) + '_stacked_calibrated_histogram.pdf'))
        run_canvas.Print((path + str(group_data[0].run_number) + '_stacked_calibrated_histogram.pdf]'))
    else:
        print('Plotting calibrated energy histogram.')
        run_canvas = TCanvas("run_canvas", "run canvas")
        run_canvas.Divide(3, int(len(cal_pars_list) / 2))
        canvas_index = 1
        for entry in hists_and_lines:
            pad = run_canvas.cd(canvas_index)
            pad.SetLogy()
            entry[0].Draw()
            entry[1].Draw()
            entry[2].Draw()
            pad.Update()
            canvas_index += 1

            path = 'output/group_' + group_name + '/'
            run_canvas.Print((path + name + '_calibrated_histograms.pdf['))
            run_canvas.Print((path + name + '_calibrated_histograms.pdf'))
            run_canvas.Print((path + name + '_calibrated_histograms.pdf]'))


def plot_temp_data(temp_data):
    """
    Plots a line plot of temperature versus time.
    """
    times = temp_data.times
    temperatures = temp_data.temperatures
    run_number = str(temp_data.run_number)
    print('Plotting background temperature data.')
    fig, ax = plt.subplots()
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature ($^{\circ}$C)')
    ax.set_title('Temperature vs Time')

    secs = mdates.epoch2num(times)
    ax.plot(secs, temperatures)
    date_fmt = '%m-%d %H:%M:%S'
    date_formatter = mdates.DateFormatter(date_fmt)
    ax.xaxis.set_major_formatter(date_formatter)

    print('Writing background temperature data to file.')
    temp_array = temp_array = np.array(temperatures)
    filename = 'output/group_' + run_number + '/' + 'temperature_output.txt'
    outfile = open(filename, 'w')
    outfile.write('Run:\t\t\t' + run_number + '\n')
    outfile.write('Start time:\t\t' + time.ctime(times[0]) + '\n')
    outfile.write('Start temperature:\t' + str(temperatures[0]) + '\n')
    outfile.write('\n')
    outfile.write('Stop time:\t\t' + time.ctime(times[-1]) + '\n')
    outfile.write('Stop temperature:\t' + str(temperatures[-1]) + '\n')
    outfile.write('\n')
    outfile.write('Highest temperature:\t' + str(temp_array.max()) + '\n')
    outfile.write('Lowest temperature:\t' + str(temp_array.min()) + '\n')
    outfile.write('\n')
    outfile.write('Average:\t\t' + str(np.mean(temp_array)) + '\n')
    outfile.write('Variance:\t\t' + str(np.var(temp_array)) + '\n')
    outfile.write('Standard Deviation:\t' + str(np.std(temp_array)) + '\n')
    outfile.close()
    fig.autofmt_xdate()
    plt.savefig('output/group_%s/group_%s_temperature_plot.pdf' % (run_number, run_number))


def plot_correlations(group):
    K_counts_data = []
    Tl_counts_data = []
    K_mu_data = []
    Tl_mu_data = []
    K_sigma_data = []
    Tl_sigma_data = []
    temp_data = []
    time_data = []
    group_name = group[0]
    for run in group:
        filename = 'output/group_%s/%s_output.txt' % (group_name, run)
        temp_data_file = open(filename, 'r')
        lines = temp_data_file.readlines()
        start_time = float(lines[1].split('\t')[-1])
        time_data.append(start_time)

        average_temperature = float(lines[9].split('\t')[-1])
        temp_data.append(average_temperature)

        K_counts = float(lines[14].split('\t')[-1])
        K_counts_data.append(K_counts)
        Tl_counts = float(lines[21].split('\t')[-1])
        Tl_counts_data.append(Tl_counts)

        K_mu = float(lines[15].split('\t')[-1])
        K_mu_data.append(abs(K_mu))
        Tl_mu = float(lines[22].split('\t')[-1])
        Tl_mu_data.append(abs(Tl_mu))

        K_sigma = float(lines[16].split('\t')[-1])
        K_sigma_data.append(abs(K_sigma))
        Tl_sigma = float(lines[23].split('\t')[-1])
        Tl_sigma_data.append(abs(Tl_sigma))
        temp_data_file.close()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, figsize=(8, 6))
    secs = mdates.epoch2num(time_data)
    major_date_formatter = mdates.DateFormatter('%m-%d %H:%M')

    print('Plotting Temperature vs Time.')
    ax1.set_title('Temperature vs Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Temperature ($^{\circ}$C)')
    #ax1.set_ylim(0, 25)
    ax1.plot(secs, temp_data, '-o')
    ax1.xaxis.set_major_formatter(major_date_formatter)
    ax1.legend(['Temperature'], loc='upper right', fontsize='x-small')

    print('Plotting Peak Position vs Time.')
    corrected_K_mu = np.array(K_mu_data)
    K_mu_average = np.mean(corrected_K_mu)
    corrected_K_mu -= K_mu_average
    corrected_Tl_mu = np.array(Tl_mu_data)
    Tl_mu_average = np.mean(corrected_Tl_mu)
    corrected_Tl_mu -= Tl_mu_average
    ax2.set_title('Peak Position vs Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Uncalibrated Energy (ADC)')
    ax2.plot(secs, corrected_K_mu, '-o', secs, corrected_Tl_mu, '-o')
    ax2.xaxis.set_major_formatter(major_date_formatter)
    ax2.legend(['K peak', 'Tl peak'], loc='upper right', fontsize='x-small')

    print('Plotting Peak Resolution vs Time.')
    corrected_K_sigma = np.array(K_sigma_data)
    K_sigma_average = np.mean(corrected_K_sigma)
    corrected_K_sigma -= K_sigma_average
    corrected_Tl_sigma = np.array(Tl_sigma_data)
    Tl_sigma_average = np.mean(corrected_Tl_sigma)
    corrected_Tl_sigma -= Tl_sigma_average
    ax3.set_title('Peak Resolution vs Time')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Uncalibrated Energy (ADC)')
    ax3.plot(secs, corrected_K_sigma, '-o', secs, corrected_Tl_sigma, '-o')
    ax3.xaxis.set_major_formatter(major_date_formatter)
    ax3.legend(['K peak', 'Tl peak'], loc='upper right', fontsize='x-small')

    print('Plotting Counts vs Time.')
    corrected_K_counts = np.array(K_counts_data)
    K_counts_average = np.mean(corrected_K_counts)
    corrected_K_counts -= K_counts_average
    corrected_Tl_counts = np.array(Tl_counts_data)
    Tl_counts_average = np.mean(corrected_Tl_counts)
    corrected_Tl_counts -= Tl_counts_average
    ax4.set_title('Counts vs Time')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Counts')
    ax4.plot(secs, corrected_K_counts, '-o', secs, corrected_Tl_counts, '-o')
    ax4.xaxis.set_major_formatter(major_date_formatter)
    ax4.legend(['K peak', 'Tl peak'], loc='upper right', fontsize='x-small')
    fig.tight_layout(pad=2.5, w_pad=0.5, h_pad=1.5)
    fig.autofmt_xdate()
    time_distination = 'output/group_%s/%s_temp_plots.pdf' % (group_name, group_name)
    plt.savefig(time_distination)
    print('Figure saved to ', time_distination)
    plt.clf()


    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 8))

    print('Plotting Counts vs Temperature.')
    counts = sorted(zip(*[temp_data, K_counts_data, Tl_counts_data]))
    sorted_temps1, sorted_K_counts, sorted_Tl_counts = list(zip(*counts))
    corrected_K_counts = np.array(sorted_K_counts)
    K_counts_average = np.mean(corrected_K_counts)
    corrected_K_counts -= K_counts_average
    corrected_Tl_counts = np.array(sorted_Tl_counts)
    Tl_counts_average = np.mean(corrected_Tl_counts)
    corrected_Tl_counts -= Tl_counts_average
    ax1.set_title('Counts vs Temperature')
    ax1.set_xlabel('Temperature ($^{\circ}$C)')
    ax1.set_ylabel('Counts')
    ax1.plot(sorted_temps1, corrected_K_counts, '-o', sorted_temps1, corrected_Tl_counts, '-o')
    ax1.legend(['K peak', 'Tl peak'], loc='upper right', fontsize='x-small')

    print('Plotting Peak Position vs Temperature.')
    positions = sorted(zip(*[temp_data, K_mu_data, Tl_mu_data]))
    sorted_temps2, sorted_K_positions, sorted_Tl_positions = list(zip(*positions))
    corrected_K_positions = np.array(sorted_K_positions)
    K_positions_average = np.mean(corrected_K_positions)
    corrected_K_positions -= K_positions_average
    corrected_Tl_positions = np.array(sorted_Tl_positions)
    Tl_positions_average = np.mean(corrected_Tl_positions)
    corrected_Tl_positions -= Tl_positions_average
    ax2.set_title('Peak Position vs Temperature')
    ax2.set_xlabel('Temperature ($^{\circ}$C)')
    ax2.set_ylabel('Uncalibrated Energy (ADC)')
    ax2.plot(sorted_temps2, corrected_K_positions, '-o', sorted_temps2, corrected_Tl_positions, '-o')
    ax2.legend(['K peak', 'Tl peak'], loc='upper right', fontsize='x-small')

    print('Plotting Peak Resolution vs Temperature.')
    res = sorted(zip(*[temp_data, K_sigma_data, Tl_sigma_data]))
    sorted_temps3, sorted_K_res, sorted_Tl_res = list(zip(*res))
    corrected_K_res = np.array(sorted_K_res)
    K_res_average = np.mean(corrected_K_res)
    corrected_K_res -= K_res_average
    corrected_Tl_res = np.array(sorted_Tl_res)
    Tl_res_average = np.mean(corrected_Tl_res)
    corrected_Tl_res -= Tl_res_average
    ax3.set_title('Peak Resolution vs Temperature')
    ax3.set_xlabel('Temperature ($^{\circ}$C)')
    ax3.set_ylabel('Uncalibrated Energy (ADC)')
    ax3.plot(sorted_temps3, corrected_K_res, '-o', sorted_temps3, corrected_Tl_res, '-o')
    ax3.legend(['K peak', 'Tl peak'], loc='upper right', fontsize='x-small')

    fig.tight_layout(pad=2.5, w_pad=0.5, h_pad=1.5)
    fig.autofmt_xdate()
    temperature_destination = 'output/group_%s/%s_vs_temp_plots.pdf' % (group_name, group_name)
    plt.savefig(temperature_destination)
    print('Figure saved to ', temperature_destination)


def plot_gain_curves(group_data, cal_pars_list):
    group_name = str(group_data[0].run_number)
    calibrations_list = []
    voltages = [600, 700, 800, 900, 1000]
    for cal_pars in cal_pars_list:
        gains = []
        pars = parameters.gain_pars[cal_pars]
        for voltage in voltages:
            cal = pars[0] + pars[1] * voltage + pars[2] * (voltage ** 2)
            gains.append(cal)
        calibrations_list.append(gains)

    print('Plotting gain curves.')
    plt.title('Gain Curves')
    plt.xlabel('Voltages')
    plt.ylabel('Gain')
    for gains in calibrations_list:
        plt.plot(voltages, gains, '-o')
        #plt.legend(['Temperature'], loc='upper right', fontsize='x-small')
    gain_destination = 'output/group_%s/%s_log_gain_plots.pdf' % (group_name, group_name)
    plt.savefig(gain_destination)
    print('Figure saved to ', gain_destination)



if __name__ == "__main__":
    main()
