"""Select eyes-closed segments from the EEG LEMON data

"""

# Install dependencies
import os
import glob
import mne
import numpy as np
from sys import argv
from osl_dynamics.data import Data

# Set hyperparameters
if len(argv) != 2:
    print("Need to pass one argument: use of structurals (e.g., python script.py subject)")
    exit()
structurals = argv[1] # type of structurals to use
print(f"[INFO] Structurals: {structurals}")

# Define custom functions
def lemon_make_task_regressor(dataset):
    ev, ev_id = mne.events_from_annotations(dataset['raw'])
    print('Found {0} events in raw'.format(ev.shape[0]))
    print(ev_id)

    # Correct for cropping first 10 seconds - not sure why this is necessary?!
    ev[:, 0] -= dataset['raw'].first_samp

    task = np.zeros((dataset['raw'].n_times,))
    for ii in range(ev.shape[0]):
        if ev[ii, 2] == ev_id['Stimulus/S200']:
            # EYES OPEN
            task[ev[ii,0]:ev[ii,0]+5000] = 1
        elif ev[ii, 2] == ev_id['Stimulus/S210']:
            # EYES CLOSED
            task[ev[ii,0]:ev[ii,0]+5000] = -1
        elif ev[ii, 2] == 1:
            task[ev[ii,0]] = task[ev[ii,0]-1]

    return task

def reject_by_annotation(raw, data):
    # Get available channels
    picks = mne.io.pick._picks_to_idx(raw.info, None, 'all', exclude=())
    picks = np.atleast_1d(np.arange(raw.info['nchan'])[picks]) # convert to ints
    # Handle starting and stopping time points
    tmin_start, tmax_stop = raw._handle_tmin_tmax(None, None)
    # Truncate start and stop to the open interval [0, n_times]
    start = min(max(0, tmin_start), raw.n_times)
    stop = min(max(0, tmax_stop), raw.n_times)
    output = data
    if len(raw.annotations) > 0:
        onsets, ends = mne.annotations._annotations_starts_stops(raw, ['BAD'])
        keep = (onsets < stop) & (ends > start)
        onsets = np.maximum(onsets[keep], start)
        ends = np.minimum(ends[keep], stop)
        if len(onsets) == 0:
            _, times = raw[picks, start:stop]
            output = data
        else:
            n_samples = stop - start # total number of samples
            used = np.ones(n_samples, bool)
            for onset, end in zip(onsets, ends):
                if onset >= end:
                    continue
                used[onset - start:end - start] = False
            used = np.concatenate([[False], used, [False]])
            starts = np.where(~used[:-1] & used[1:])[0] + start
            stops = np.where(used[:-1] & ~used[1:])[0] + start
            n_kept = (stops - starts).sum()  # kept samples
            n_rejected = n_samples - n_kept  # rejected samples
            if n_rejected > 0:
                output = np.zeros((n_kept,))
                times = np.zeros(len(output))
                idx = 0
                for start, stop in zip(starts, stops):  # get the data
                    if start == stop:
                        continue
                    end = idx + stop - start
                    _, times[idx:end] = raw[picks, start:stop]
                    output[idx:end] = data[start:stop]
                    idx = end
            else:
                _, times = raw[picks, start:stop]
                output = data
    print("Shape of data before bad segment rejection: ", np.shape(data))
    print("Shape of data after bad segement rejection: ", np.shape(output))
    if len(np.shape(data)) != len(np.shape(output)):
        raise ValueError("Dimensions of input and output data do not match.")
    return output, times

# SELECT EYES-CLOSED SEGMENTS
if __name__ == "__main__":
    # Set up directories
    BASE_DIR = "/well/woolrich/projects/lemon/scho23"
    if structurals == "subject":
        PREPROC_DIR = BASE_DIR + "/preproc_ec"
        SRC_DIR = BASE_DIR + "/src_ec"
    elif structurals == "standard":
        PREPROC_DIR = BASE_DIR + "/preproc"
        SRC_DIR = BASE_DIR + "/src_ec_no_struct"

    # Get file paths
    preproc_file_names = sorted(glob.glob(os.path.join(PREPROC_DIR, "*/*_preproc_raw.fif")))
    print("# of subjects (preprocessed): ", len(preproc_file_names))
    src_file_names = sorted(glob.glob(os.path.join(SRC_DIR, "*/sflip_parc-raw.fif")))
    print("# of subjects (source reconstructed): ", len(src_file_names))

    # Get subject IDs
    src_subj_ids = [fname.split('/')[-2] for fname in src_file_names]

    # Only keep preprocessed files that have corresponding source reconstructed files
    preproc_file_names = [fname for fname in preproc_file_names if fname.split('/')[-1].split('_')[0] in src_subj_ids]
    if len(preproc_file_names) != len(src_file_names):
        raise ValueError("different subjects in `preproc` and `src`.")
    print("# of subjects (preprocessed) after matching: ", len(preproc_file_names))

    # Get eyes closed time points
    tasks, no_event_id = [], []
    for fname in preproc_file_names:
        raw = mne.io.read_raw_fif(fname, preload=True)
        try:
            task_raw = lemon_make_task_regressor({"raw": raw})
            task, _ = reject_by_annotation(raw, task_raw)
        except:
            task = []
            no_event_id.append(fname.split('/')[-1].split('_')[0])
        tasks.append(task)
        raw.close()

    print("Subjects with no event markings (n={}): {}".format(len(no_event_id), no_event_id))

    # Exclude subjects with no event/stimulus markings
    elim_empty = lambda X: [x for x in X if len(x) > 0]
    tasks = elim_empty(tasks)
    src_subj_ids = [id for id in src_subj_ids if id not in no_event_id]
    print("Final # of subjects to extract task info from: ", len(src_subj_ids))

    # Select eyes closed segments
    # [1] Preprocessed
    if structurals == "subject":
        print("*** Processing preprocessed data ***")
        preproc_file_names = [os.path.join(PREPROC_DIR, f"{id}/{id}_preproc_raw.fif") for id in src_subj_ids]
        preproc_data = Data(preproc_file_names, picks="eeg", reject_by_annotation="omit")
        for n, data in enumerate(preproc_data.arrays):
            print("Data length (before): ", len(data))
            print("Task length: ", len(tasks[n]))
            assert len(data) == len(tasks[n]), "Data and event time courses do not match in length."
            eyes_closed = tasks[n] == -1
            updated_data = data[eyes_closed, :]
            print("Data length (after): ", len(updated_data))
            new_name = preproc_file_names[n].replace(".fif", ".npy")
            print(f"Saving {new_name} ...")
            np.save(new_name, updated_data)
        preproc_data.delete_dir() # clean up

    # [2] Source reconstruction
    print("*** Processing source reconstructed data ***")
    src_file_names = [os.path.join(SRC_DIR, f"{id}/sflip_parc-raw.fif") for id in src_subj_ids]
    src_data = Data(src_file_names, picks="misc", reject_by_annotation="omit")
    for n, data in enumerate(src_data.arrays):
        print("Data length: ", len(data))
        print("Task length: ", len(tasks[n]))
        assert len(data) == len(tasks[n]), "Data and event time courses do not match in length."
        eyes_closed = tasks[n] == -1
        updated_data = data[eyes_closed, :]
        print("Data length (after): ", len(updated_data))
        new_name = src_file_names[n].replace(".fif", ".npy")
        print(f"Saving {new_name} ...")
        np.save(new_name, updated_data)
    src_data.delete_dir() # clean up

    print("Process Complete.")
