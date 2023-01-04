import copy
import os
import pickle
import nibabel as nib
import copy

from dipy.segment.mask import applymask
from dipy.io.image import load_nifti
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration

def inverseTransformAtlas(folder_path, p, atlasPath, atlasName, DWI_type="AP"):
    preproc_folder = folder_path + '/subjects/' + p + '/dMRI/preproc/'
    reg_path = folder_path + '/subjects/' + p + '/reg/'
    if os.path.exists(reg_path + 'mapping_T1w_to_T1wCommonSpace.p'):
        with open(reg_path + 'mapping_T1w_to_T1wCommonSpace.p', 'rb') as handle:
            mapping_T1w_to_T1wCommonSpace = pickle.load(handle)
    else:
        raise Exception("No mapping_T1w_to_T1wCommonSpace.p file found in " + reg_path)
        return

    DWI_subject = preproc_folder + p + "_dmri_preproc.nii.gz"
    AP_subject = folder_path + '/subjects/' + p + '/masks/' + p + '_ap.nii.gz'
    WM_FOD_subject = folder_path + '/subjects/' + p + '/dMRI/ODF/MSMT-CSD/' + p + "_MSMT-CSD_WM_ODF.nii.gz"

    if DWI_type == "AP" and os.path.exists(reg_path + 'mapping_DWI_AP_to_T1.p') and os.path.exists(AP_subject):
        with open(reg_path + 'mapping_DWI_AP_to_T1.p', 'rb') as handle:
            mapping_DWI_to_T1 = pickle.load(handle)
        subject_map = nib.load(AP_subject)
    elif DWI_type == "B0" and os.path.exists(reg_path + 'mapping_DWI_B0_to_T1.p') and os.path.exists(DWI_subject):
        with open(reg_path + 'mapping_DWI_B0_to_T1.p', 'rb') as handle:
            mapping_DWI_to_T1 = pickle.load(handle)
        subject_map = nib.load(DWI_subject)
    elif DWI_type == "WMFOD" and os.path.exists(reg_path + 'mapping_DWI_WMFOD_to_T1.p') and os.path.exists(WM_FOD_subject):
        with open(reg_path + 'mapping_DWI_WMFOD_to_T1.p', 'rb') as handle:
            mapping_DWI_to_T1 = pickle.load(handle)
        subject_map = nib.load(WM_FOD_subject)
    else:
        raise Exception("No mapping_DWI_to_T1.p file found in " + reg_path)
        return

    atlas = nib.load(atlasPath)
    atlas_data = atlas.get_fdata()
    atlas_data_T1space = mapping_T1w_to_T1wCommonSpace.transform_inverse(atlas_data)
    atlas_data_DWIspace = mapping_DWI_to_T1.transform_inverse(atlas_data_T1space)

    atlasProjectedHeader = copy.deepcopy(atlas.header)
    atlasProjectedHeader["dim"][1:4] = atlas_data_DWIspace.shape[0:3]
    atlasProjectedHeader["pixdim"] = subject_map.header["pixdim"]

    out_DWI = nib.Nifti1Image(atlas_data_DWIspace, subject_map.affine, atlasProjectedHeader)
    out_DWI.to_filename(reg_path + p + "_Atlas_" + atlasName + "_InSubjectDWISpaceFrom_" + DWI_type + ".nii.gz")

    #out_T1 = nib.Nifti1Image(atlas_data_T1space, None, None)
    #out_T1.to_filename(reg_path + p + "_Atlas_" + atlasName + "_InSubjectT1Space.nii.gz")

