import elikopy
import os
import subprocess
import nibabel as nib
import numpy as np
from dipy.io.image import load_nifti, save_nifti
import dipy.tracking
from dipy.io.streamline import load_tractogram

def connectivityMatrix(folder_path, p, label_fname):
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
    tractogram_path = tracking_path + p + '_tractogram.tck'
    if os.path.exists(tractogram_path):
        tractogram = load_tractogram(tractogram_path, "same")

    MV, grouping = dipy.tracking.utils.connectivity_matrix(tractogram.streamlines, img.affine, labels,
                                             return_mapping=True,
                                             mapping_as_streamlines=True)

    MV[:3, :] = 0
    MV[:, :3] = 0


    import matplotlib.pyplot as plt
    np.save(tracking_path + p + "_connectivity_matrix.npy", MV)
    plt.imshow(np.log1p(MV), interpolation='nearest')
    plt.savefig(tracking_path + "connectivity.png")
