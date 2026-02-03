import elikopy
import os
import subprocess
import nibabel as nib
import numpy as np
from dipy.io.image import load_nifti, save_nifti
import dipy.tracking
from dipy.io.streamline import load_tractogram
from scipy.ndimage import binary_dilation
from skimage.morphology import ball
from collections import defaultdict
from dipy.tracking._utils import _mapping_to_voxel, _to_voxel_coordinates
from nibabel.streamlines import ArraySequence as Streamlines
from itertools import combinations

def load_sift2_weights(weights_file: str, mu_file: str):

    text = []

    f = open(weights_file, "r")
    for x in f:
        text.append(x)
    f.close()
    
    f = open(mu_file, "r")
    for x in f:
        mu=float(x)
    f.close()

    w = np.array([float(i) for i in text[1].split(' ')], dtype='float32')
    
    return w,mu


def dipy_connectivity_matrix(
    streamlines,
    affine,
    label_volume,
    *,
    inclusive=False,
    symmetric=True,
    weights=None,
    discard_stream_size=0,
    return_mapping=False,
    mapping_as_streamlines=False,
):
    """Count the streamlines that start and end at each label pair.

    Parameters
    ----------
    streamlines : sequence
        A sequence of streamlines.
    affine : array_like (4, 4)
        The mapping from voxel coordinates to streamline coordinates.
        The voxel_to_rasmm matrix, typically from a NIFTI file.
    label_volume : ndarray
        An image volume with an integer data type, where the intensities in the
        volume map to anatomical structures.
    inclusive: bool
        Whether to analyze the entire streamline, as opposed to just the
        endpoints.
    symmetric : bool, optional
        Symmetric means we don't distinguish between start and end points. If
        symmetric is True, ``matrix[i, j] == matrix[j, i]``.
    weights : ndarray, optional
        A 1D array of size n, containing the weights of each of the n
        streamlines.
    discard_stream_size : int, optional
        If the length of a streamline is less than or equal to this value, it
        will not be included in the connectivity matrix. When 0, no filtering
        is applied. This is useful for ignoring very short streamlines that
        are likely to be noise.
    return_mapping : bool, optional
        If True, a mapping is returned which maps matrix indices to
        streamlines.
    mapping_as_streamlines : bool, optional
        If True voxel indices map to lists of streamline objects. Otherwise
        voxel indices map to lists of integers.

    Returns
    -------
    matrix : ndarray
        The number of connection between each pair of regions in
        `label_volume`.
    mapping : defaultdict(list)
        ``mapping[i, j]`` returns all the streamlines that connect region `i`
        to region `j`. If `symmetric` is True mapping will only have one key
        for each start end pair such that if ``i < j`` mapping will have key
        ``(i, j)`` but not key ``(j, i)``.
    """

    # Error checking on label_volume
    kind = label_volume.dtype.kind
    labels_positive = (kind == "u") or ((kind == "i") and (label_volume.min() >= 0))
    valid_label_volume = labels_positive and label_volume.ndim == 3
    if not valid_label_volume:
        raise ValueError(
            "label_volume must be a 3d integer array with non-negative label values"
        )

    mapping = defaultdict(list)
    lin_T, offset = _mapping_to_voxel(affine)

    if type(streamlines).__name__ == "generator":
        streamlines = Streamlines(streamlines)

    if weights is None:
        weights = np.ones(len(streamlines))
        matrix = np.zeros(
            (np.max(label_volume) + 1, np.max(label_volume) + 1), dtype=np.int64
        )
    else:
        matrix = np.zeros((np.max(label_volume) + 1, np.max(label_volume) + 1))

    if discard_stream_size > 0:
        (keep_idx,) = np.where(streamlines._lengths > discard_stream_size)
        streamlines = streamlines[keep_idx]
        weights = weights[keep_idx]

    if inclusive:
        for i, sl in enumerate(streamlines):
            sl = _to_voxel_coordinates(sl, lin_T, offset)
            x, y, z = sl.T
            if symmetric:
                crossed_labels = np.unique(label_volume[x, y, z])
            else:
                crossed_labels = np.unique(label_volume[x, y, z], return_index=True)
                crossed_labels = crossed_labels[0][np.argsort(crossed_labels[1])]

            for comb in combinations(crossed_labels, 2):
                matrix[comb] += weights[i]

                if return_mapping:
                    if mapping_as_streamlines:
                        mapping[comb].append(streamlines[i])
                    else:
                        mapping[comb].append(i)

    else:
        streamlines_end = np.array([sl[[0, -1]] for sl in streamlines])
        streamlines_end = _to_voxel_coordinates(streamlines_end, lin_T, offset)
        x, y, z = streamlines_end.T
        if symmetric:
            end_labels = np.sort(label_volume[x, y, z], axis=0)
        else:
            end_labels = label_volume[x, y, z]
        np.add.at(matrix, (end_labels[0].T, end_labels[1].T), weights)

        if return_mapping:
            if mapping_as_streamlines:
                for i, (a, b) in enumerate(end_labels.T):
                    mapping[a, b].append(streamlines[i])
            else:
                for i, (a, b) in enumerate(end_labels.T):
                    mapping[a, b].append(i)

    if symmetric:
        matrix = np.maximum(matrix, matrix.T)

    if return_mapping:
        return (matrix, mapping)
    else:
        return matrix


def connectivityMatrix(folder_path, p, label_fname, input="TCKGEN", inclusive=False, dilation_radius=0, longitudinal=False, tractogram_filename="tractogram", suffix=""):

    assert input in ["TCKGEN", "SIFT", "SIFT2"], "input must be either TCKGEN, SIFT or SIFT2"

    dwi_path = folder_path + '/subjects/' + p + '/dMRI/preproc/' + p + '_dmri_preproc.nii.gz'

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
    if longitudinal:
        label_fpath = os.path.join(reg_path, p + "_Atlas_" + label_fname + "_longitudinal.nii.gz")
    else:
        label_fpath = os.path.join(reg_path, p + "_Atlas_" + label_fname + ".nii.gz")
    labels_nii = nib.load(label_fpath)
    labels = np.round(labels_nii.get_fdata()).astype(int)
    
    tracking_path = folder_path + '/subjects/' + p + "/dMRI/tractography/"
    
    if dilation_radius>0:
        # Create a spherical structuring element for dilation
        #struct_el = ball(dilation_radius)
        
        n_dim = len(labels.shape)
        struct_el = np.zeros((3,)*n_dim)
        for d in range(n_dim):
            idc = tuple([slice(None) if d == i else 1 for i in range(n_dim)])
            struct_el[idc] = 1

        # Dilate each label separately (assuming labels are integer values starting from 1)
        dilated_label_data = np.zeros_like(labels)
        solo_labels = np.unique(labels)[1:]  # Exclude background (0)
        for label in solo_labels:
            mask = labels == label
            dilated_mask = binary_dilation(mask, structure=struct_el)
            dilated_label_data[dilated_mask] = label
        
        for label in solo_labels:
            mask = labels == label
            dilated_label_data[mask] = label
        
        labels = np.round(dilated_label_data).astype(int)
        print("Shape:", dilated_label_data.shape)
        dilated_label_img = nib.Nifti1Image(dilated_label_data, labels_nii.affine, dtype=np.uint16)
        nib.save(dilated_label_img, tracking_path + f'dilated_{dilation_radius}_labels.nii.gz')

    
    if input == "TCKGEN":
        tractogram_path = tracking_path + p + f'_{tractogram_filename}.tck'
        weights = None
    elif input == "SIFT":
        tractogram_path = tracking_path + p + f'_{tractogram_filename}_sift.tck'
        weights = None
    elif input == "SIFT2":
        tractogram_path = tracking_path + p + f'_{tractogram_filename}.tck'
        weights_file = tracking_path + p + f'_{tractogram_filename}_sift2.txt'
        mu_file = tracking_path + p + f'_{tractogram_filename}_sift2_mu.txt'
        weights,mu = load_sift2_weights(weights_file, mu_file)
        weights = weights*mu
    else:
        raise Exception("Invalid input type")

    if os.path.exists(tractogram_path):
        tractogram = load_tractogram(tractogram_path, dwi_path)
    else:
        raise Exception("No tractogram found in " + tracking_path)
        return

    # Compute the connectivity matrix
    MV, grouping = dipy_connectivity_matrix(tractogram.streamlines, img.affine,
                                            labels, inclusive=inclusive,
                                            return_mapping=True,
                                            mapping_as_streamlines=True,
                                            weights = weights)

    MV = np.delete(MV, 0, 0)
    MV = np.delete(MV, 0, 1)


    import matplotlib.pyplot as plt
    plt.imshow(np.log1p(MV), interpolation='nearest')

    if longitudinal:
        np.save(tracking_path + f"{p}_type-{input}_atlas-{label_fname}_inclusive-{inclusive}_dilate-{dilation_radius}_time-longitudinal_connectivityMatrix{suffix}.npy", MV)
        plt.savefig(tracking_path + f"{p}_type-{input}_atlas-{label_fname}_inclusive-{inclusive}_dilate-{dilation_radius}_time-longitudinal_connectivityMatrix{suffix}.png")
    else:
        np.save(tracking_path + f"{p}_type-{input}_atlas-{label_fname}_inclusive-{inclusive}_dilate-{dilation_radius}_connectivityMatrix{suffix}.npy", MV)
        plt.savefig(tracking_path + f"{p}_type-{input}_atlas-{label_fname}_inclusive-{inclusive}_dilate-{dilation_radius}_connectivityMatrix{suffix}.png")
