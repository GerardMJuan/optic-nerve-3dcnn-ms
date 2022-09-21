
import nibabel as nib
import numpy as np

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

def normalize(volume):
    """Normalize the volume"""
    min = np.amin(volume) # HARDCODED ( need to change )
    max = np.amax(volume) # HARDCODED ( need to change )
    #volume[volume < min] = min
    #volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def normalize_mean(volume):
    mean = np.average(volume)
    std = np.std(volume)
    volume = (volume - mean) / std
    volume = volume.astype("float32")
    return volume

def process_scan(path, filename):
    """Read and resize volume
    
    TODO: CHANGE THIS FUNCTION SO THAT IT RETURNS A LIST OF SCANS"""
    # Read scan
    volume_list = []
    # WE HAVE 9 SCANS PER SUBJECT
    for i in range(1,10):
        # todo: create path for the speficic image
        new_path = f'{path}/Eye_{i}_{filename}'
        volume = read_nifti_file(new_path)
        # Normalize
        #volume = normalize(volume)

        # need to apply normalize mean to all volumes
        volume = normalize_mean(volume)
        # Resize width, height and depth
        # volume = resize_volume(volume)
        volume_list.append(volume)
    return volume_list

def process_scan_old(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    #volume = normalize(volume)
    volume = normalize_mean(volume)
    # Resize width, height and depth
    # volume = resize_volume(volume)
    return volume