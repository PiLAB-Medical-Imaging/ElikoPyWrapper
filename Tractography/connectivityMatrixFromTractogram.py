import elikopy
import os
import subprocess
import nibabel as nib
import numpy as np
from dipy.io.image import load_nifti, save_nifti
import dipy.tracking
from dipy.io.streamline import load_tractogram

def connectivityMatrix(folder_path, p, label_fname, input="TCKGEN"):

    assert input in ["TCKGEN", "SIFT", "SIFT2"], "input must be either TCKGEN, SIFT or SIFT2"
    
    odf_wmfod = folder_path + '/subjects/' + p + "/dMRI/ODF/MSMT-CSD/" + p + "_MSMT-CSD_WM_ODF.nii.gz"
    odf_csd_mrtrix = folder_path + '/subjects/' + p + "/dMRI/ODF/CSD/" + p + "_CSD_SH_ODF_mrtrix.nii.gz"
    if os.path.exists(odf_wmfod):
        img = nib.load(odf_wmfod)
    elif os.path.exists(odf_csd_mrtrix):
        img = nib.load(odf_csd_mrtrix)
    else:
        raise Exception("No ODF file found in " + folder_path + '/subjects/' + p + "/dMRI/ODF/")
        return

    reg_path = folder_path + '/subjects/' + p + '/reg/'
    label_fpath = os.path.join(reg_path, p + "_Atlas_" + label_fname + ".nii.gz")
    labels_nii = nib.load(label_fpath)
    labels = np.round(labels_nii.get_fdata()).astype(int)

    tracking_path = folder_path + '/subjects/' + p + "/dMRI/tractography/"
    if input == "TCKGEN":
        tractogram_path = tracking_path + p + '_tractogram.trk'
    elif input == "SIFT":
        tractogram_path = tracking_path + p + '_tractogram_sift.tck'
    elif input == "SIFT2":
        raise Exception("SIFT2 not implemented yet")
        pass # TODO
    else:
        raise Exception("Invalid input type")

    if os.path.exists(tractogram_path):
        tractogram = load_tractogram(tractogram_path, "same")
    else:
        raise Exception("No tractogram found in " + tracking_path)
        return

    # Compute the connectivity matrix
    MV, grouping = dipy.tracking.utils.connectivity_matrix(tractogram.streamlines, img.affine, labels,
                                             return_mapping=True,
                                             mapping_as_streamlines=True)

    MV = np.delete(MV, 0, 0)
    MV = np.delete(MV, 0, 1)


    import matplotlib.pyplot as plt
    plt.imshow(np.log1p(MV), interpolation='nearest')

    if input == "SIFT" or input == "SIFT2":
        np.save(tracking_path + p + "_connectivity_matrix_" + input + ".npy", MV)
        plt.savefig(tracking_path + "connectivity" + input + ".png")
