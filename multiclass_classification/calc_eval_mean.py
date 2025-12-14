import sys
import numpy as np

# Simple hardcoded string manipulation to extract accuracy and ece values from log file
def calc_metrics(logfile):
    
    with open(logfile) as f:
        lines = f.readlines()

        metrics = {}

        # Acc@1
        metrics['acc1'] = float(lines[-2].split()[4].split('%')[0])
        # Acc@5
        metrics['acc5'] = float(lines[-2].split()[6].split('%')[0])
        # HAcc
        metrics['hacc'] = float(lines[-2].split()[8].split('%')[0])
        # MAcc
        metrics['macc'] = float(lines[-2].split()[10].split('%')[0])
        # TAcc
        metrics['tacc'] = float(lines[-2].split()[12].split('%')[0])
        # Ece
        metrics['ece'] = float(lines[-1].split()[-1].split('%')[0])

        return metrics



def main():
    if len(sys.argv) != 2:
        raise AssertionError("Only supply txt with eval directories to include!")
    metrics = {}
    metrics['acc1'] = []
    metrics['acc5'] = []
    metrics['hacc'] = []
    metrics['macc'] = []
    metrics['tacc'] = []
    metrics['ece'] = []

    # Read all logs in supplied txt 
    with open(sys.argv[1]) as f:
        lines = f.readlines()
        for logdir in lines:
            # Cut off date to get file name
            date_ind = logdir.rfind('_')
            txtname = logdir[:date_ind]
            # Path to log file
            file = f'saved/{logdir.strip()}/logs/{txtname}.txt'
            # Extract metric values
            log_vals = calc_metrics(file)
            for key in metrics:
                metrics[key].append(log_vals[key])
    # Calculate mean and std
    for key in metrics:
        ar = np.array(metrics[key])
        print(f'{key}: {np.mean(ar)} +/- {np.std(ar)}')
                


if __name__ == '__main__':
    main()