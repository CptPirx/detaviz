import matplotlib.pyplot as plt
import numpy as np


def plot_simulation_history(predicted_labels, true_labels, errors, run_times, model_path, domain, cycle_count):
    """

    :param predicted_labels:
    :param errors:
    :param run_times:
    :param model_path:
    :param threshold:
    :param domain:
    :param cycle_count:
    """
    # Calculate mean of sent samples, errors and run times
    run_times[0] = 0
    mean_errors = [np.mean(errors)] * len(errors)
    mean_time = [np.mean(run_times)] * len(run_times)

    # Plot the data
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)

    # Plot the samples
    ax[0].plot(predicted_labels, label='Predicted labels')
    ax[0].plot(true_labels, label='True labels')
    ax[0].legend(loc='upper left')
    ax[0].set(xlabel='Cycle', ylabel='Label', title='True and predicted labels')

    # # Add text box
    # text_0 = 'Mean value={mean} %'.format(mean=np.round(sent_samples_mean[0], decimals=3))
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # ax[0].text(0.7, 0.95, text_0, transform=ax[0].transAxes, fontsize=14,
    #            verticalalignment='top', bbox=props)

    # Plot the errors
    ax[1].plot(errors, label='Accuracy')
    ax[1].plot(mean_errors, label='Mean accuracy', linestyle='--')
    ax[1].legend(loc='upper left')
    ax[1].set(xlabel='Cycle', ylabel='%', title='Accuracy per cycle')

    # Add text box
    text_1 = 'Mean value={mean} %'.format(mean=(np.round(mean_errors[0], decimals=2) * 100))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax[1].text(0.7, 0.95, text_1, transform=ax[1].transAxes, fontsize=14,
               verticalalignment='top', bbox=props)

    # Plot the run time
    ax[2].plot(run_times, label='Time')
    ax[2].plot(mean_time, label='Mean time', linestyle='--')
    ax[2].legend(loc='upper left')
    ax[2].set(xlabel='Cycle', ylabel='Ms', title='Time per cycle')

    # Add text box
    text_2 = 'Mean value={mean} ms'.format(mean=np.round(mean_time[0], decimals=3))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax[2].text(0.7, 0.95, text_2, transform=ax[2].transAxes, fontsize=14,
               verticalalignment='top', bbox=props)

    # Save the figure
    fig.savefig((model_path + '/Results_cycles-{cycle_count}_sensorID-{sensor}.png').format(
            cycle_count=cycle_count,
            sensor=domain
    ),
            bbox_inches='tight')

    plt.show()