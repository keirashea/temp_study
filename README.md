# temp_study
COHERENT Crystal Characterization Temperature Study Code

temp_data.py
Run by coherent computer. Processes temperature data and gain data from root file.

take_temp.py
Run by raspberry pi. Records temperature to local txt file and uses scp to transfer temperture data back
to coherent computer.
  -> auto_process.py (not on this repository)
     Run by coherent computer. Sends command to raspberry pi to start taking temperature data.
