import copy
import os
import pickle
import nibabel as nib
import copy
import numpy as np
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

def inverseTransformAtlas(folder_path, p, atlasPath, atlasName, DWI_type="AP", longitudinal=False):
    preproc_folder = folder_path + '/subjects/' + p + '/dMRI/preproc/'
    reg_path = folder_path + '/subjects/' + p + '/reg/'
    if os.path.exists(reg_path + 'mapping_T1w_to_T1wCommonSpace.p'):
        with open(reg_path + 'mapping_T1w_to_T1wCommonSpace.p', 'rb') as handle:
            mapping_T1w_to_T1wCommonSpace = pickle.load(handle)
    else:
        raise Exception("No mapping_T1w_to_T1wCommonSpace.p file found in " + reg_path)
        return

    if longitudinal:
        if os.path.exists(reg_path + 'mapping_T1w_to_T1wRef.p'):
            with open(reg_path + 'mapping_T1w_to_T1wRef.p', 'rb') as handle:
                mapping_T1w_to_T1wRef = pickle.load(handle)
        else:
            raise Exception("No mapping_T1w_to_T1wRef.p file found in " + reg_path)
            return
    else:
        mapping_T1w_to_T1wRef = None

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
        
    #help(mapping_DWI_to_T1)

    atlas = nib.load(atlasPath)
    atlas_data = atlas.get_fdata()
    atlas_data_T1space = mapping_T1w_to_T1wCommonSpace.transform_inverse(atlas_data, interpolation='nearest')
    #atlas_data_T1space = np.around(atlas_data_T1space)
    #atlas_data_T1space = atlas_data_T1space.astype(np.uint8)
    if mapping_T1w_to_T1wRef is not None:
        atlas_data_T1space = mapping_T1w_to_T1wRef.transform_inverse(atlas_data_T1space, interpolation='nearest')
    atlas_data_DWIspace = mapping_DWI_to_T1.transform_inverse(atlas_data_T1space, interpolation='nearest')

    atlasProjectedHeader = copy.deepcopy(atlas.header)
    atlasProjectedHeader["dim"][1:4] = atlas_data_DWIspace.shape[0:3]
    atlasProjectedHeader["pixdim"] = subject_map.header["pixdim"]

    atlas_data_DWIspace = np.around(atlas_data_DWIspace)
    atlas_data_DWIspace = atlas_data_DWIspace.astype(np.uint8)
    out_DWI = nib.Nifti1Image(atlas_data_DWIspace, subject_map.affine, atlasProjectedHeader)
    if longitudinal:
        out_DWI.to_filename(reg_path + p + "_Atlas_" + atlasName + "_InSubjectDWISpaceFrom_" + DWI_type + "_longitudinal.nii.gz")
    else:
        out_DWI.to_filename(reg_path + p + "_Atlas_" + atlasName + "_InSubjectDWISpaceFrom_" + DWI_type + ".nii.gz")
    
    atlasProjectedT1Header = copy.deepcopy(atlas.header)
    atlasProjectedT1Header["dim"][1:4] = atlas_data_T1space.shape[0:3]
    out_T1 = nib.Nifti1Image(atlas_data_T1space, None, atlasProjectedT1Header)
    if longitudinal:
        out_T1.to_filename(reg_path + p + "_Atlas_" + atlasName + "_InSubjectT1Space_longitudinal.nii.gz")
    else:
        out_T1.to_filename(reg_path + p + "_Atlas_" + atlasName + "_InSubjectT1Space.nii.gz")

