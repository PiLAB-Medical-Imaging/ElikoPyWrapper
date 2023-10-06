import elikopy
import os
import subprocess
from pathlib import Path
import json
from dipy.io.image import load_nifti
import nibabel as nib
import sys
from elikopy.registration import regallDWIToT1wToT1wCommonSpace, applyTransformToAllMapsInFolder
import traceback

wrapper_path = r"/auto/globalscratch/users/l/d/ldricot/ElikoPyWrapper/"
atlasreg_path= wrapper_path + r"Registration/"
sys.path.append(atlasreg_path)

connM_path= wrapper_path + r"Tractography/"
sys.path.append(connM_path)

from registerAtlasFromCommonSpaceToSubjectSpace import inverseTransformAtlas
from connectivityMatrixFromTractogram import connectivityMatrix
from siftTractography import siftComputation


T1_MNI="${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz"
T1_MNI_mask="${FSLDIR}/data/standard/MNI152_T1_1mm_brain_mask.nii.gz"
T1_MNI=r'/CECI/proj/pilab/PermeableAccess/meditant_h67jkhFfG9uyD/MNI_ICBM152_T1_NLIN_ASYM_09c_BRAIN.nii.gz'
T1_MNI_mask=None

def printError(ex_type, ex_value, ex_traceback):
    # Extract unformatter stack traces as tuples
    trace_back = traceback.extract_tb(ex_traceback)

    # Format stacktrace
    stack_trace = list()

    for trace in trace_back:
        stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))

    print("Exception type : %s " % ex_type.__name__)
    print("Exception message : %s" %ex_value)
    print("Stack trace : %s" %stack_trace, flush=True)

def processingPipeline(folder_path, p, slurm_email, singleShell=False):
    study = elikopy.core.Elikopy(folder_path, slurm=False, slurm_email=slurm_email, cuda=False)
    
    dic_path = "/home/users/q/d/qdessain/Script_python/fixed_rad_dist.mat"
    error = False
    json_status_file = os.path.join(folder_path,"subjects",p,f"{p}_status.json")
    if os.path.exists(json_status_file):
        with open(json_status_file,"r") as f:
            patient_status = json.load(f)
    else:
        patient_status = {}
        
    if not (patient_status.get('preproc') is not None and patient_status["preproc"] == True):
        try:
            print("preproc",flush=True)
            study.preproc(eddy=True,
                        topup=True,
                        forceSynb0DisCo=True,
                        denoising=True,  
                        reslice=False, 
                        gibbs=False, 
                        biasfield=False,
                        qc_reg=True,
                        starting_state=None, 
                        report=True, patient_list_m=[p])
            patient_status["preproc"] = True
        except Exception as e:
            patient_status["preproc"] = False
            error = True
            ex_type, ex_value, ex_traceback = sys.exc_info()
            printError(ex_type, ex_value, ex_traceback)
            print(e, flush=True)
            
            
            
    with open(json_status_file,"w") as f:
            json.dump(patient_status, f, indent = 6)
            
    if patient_status.get('preproc') is not None and patient_status["preproc"] == True:
        
        if not (patient_status.get('wm_mask_FSL_T1') is not None and patient_status["wm_mask_FSL_T1"] == True):
            try:
                print("wm_mask_FSL_T1",flush=True)
                study.white_mask("wm_mask_FSL_T1", corr_gibbs=True, cpus=1, debug=False, patient_list_m=[p])
                patient_status["wm_mask_FSL_T1"] = True
            except Exception as e:
                patient_status["wm_mask_FSL_T1"] = False
                error = True
                ex_type, ex_value, ex_traceback = sys.exc_info()
                printError(ex_type, ex_value, ex_traceback)
                print(e, flush=True)
            with open(json_status_file,"w") as f:
                json.dump(patient_status, f, indent = 6)
            
        if not (patient_status.get('wm_mask_AP') is not None and patient_status["wm_mask_AP"] == True):
            try:
                print("wm_mask_AP",flush=True)
                study.white_mask("wm_mask_AP", cpus=1, debug=False, patient_list_m=[p])
                patient_status["wm_mask_AP"] = True
            except Exception as e:
                patient_status["wm_mask_AP"] = False
                error = True
                ex_type, ex_value, ex_traceback = sys.exc_info()
                printError(ex_type, ex_value, ex_traceback)
                print(e, flush=True)
            with open(json_status_file,"w") as f:
                json.dump(patient_status, f, indent = 6)
            
        if not (patient_status.get('dti') is not None and patient_status["dti"] == True):
            try:
                print("dti",flush=True)
                study.dti(patient_list_m=[p])
                patient_status["dti"] = True
            except Exception as e:
                patient_status["dti"] = False
                error = True
                ex_type, ex_value, ex_traceback = sys.exc_info()
                printError(ex_type, ex_value, ex_traceback)
                print(e, flush=True)
            with open(json_status_file,"w") as f:
                json.dump(patient_status, f, indent = 6)
            
        if not (patient_status.get('odf_msmtcsd') is not None and patient_status["odf_msmtcsd"] == True):
            try:
                print("odf_msmtcsd",flush=True)
                study.odf_msmtcsd(num_peaks=2, peaks_threshold=0.25, cpus=1, patient_list_m=[p])
                patient_status["odf_msmtcsd"] = True
            except Exception as e:
                patient_status["odf_msmtcsd"] = False
                error = True
                ex_type, ex_value, ex_traceback = sys.exc_info()
                printError(ex_type, ex_value, ex_traceback)
                print(e, flush=True)
            with open(json_status_file,"w") as f:
                json.dump(patient_status, f, indent = 6)
            
        if not (patient_status.get('odf_csd') is not None and patient_status["odf_csd"] == True):
            try:
                print("odf_csd",flush=True)
                study.odf_csd(num_peaks=2, peaks_threshold=0.25, patient_list_m=[p])
                patient_status["odf_csd"] = True
            except Exception as e:
                patient_status["odf_csd"] = False
                error = True
                ex_type, ex_value, ex_traceback = sys.exc_info()
                printError(ex_type, ex_value, ex_traceback)
                print(e, flush=True)
            with open(json_status_file,"w") as f:
                json.dump(patient_status, f, indent = 6)
                
        if not (patient_status.get('noddi') is not None and patient_status["noddi"] == True):
            try:
                print("noddi",flush=True)
                study.noddi(cpus=1, patient_list_m=[p])
                patient_status["noddi"] = True
            except Exception as e:
                print(e, flush=True)
                patient_status["noddi"] = False
                error = True
            with open(json_status_file,"w") as f:
                json.dump(patient_status, f, indent = 6)
           
        if patient_status["odf_msmtcsd"] == True:
            if not (patient_status.get('fingerprinting') is not None and patient_status["fingerprinting"] == True) and not singleShell:
                try:
                    print("fingerprinting",flush=True)
                    study.fingerprinting(dictionary_path=dic_path, mfdir="mf", patient_list_m=[p], cpus=1, peaksType="MSMT-CSD")
                    patient_status["fingerprinting"] = True
                except Exception as e:
                    patient_status["fingerprinting"] = False
                    error = True
                    ex_type, ex_value, ex_traceback = sys.exc_info()
                    printError(ex_type, ex_value, ex_traceback)
                    print(e, flush=True)
                with open(json_status_file,"w") as f:
                    json.dump(patient_status, f, indent = 6)
                    
        if patient_status.get('wm_mask_AP') is not None and patient_status.get('dti') is not None and patient_status["wm_mask_AP"] == True and patient_status["dti"] == True:
            if not (patient_status.get('regallDWIToT1wToT1wCommonSpace') is not None and patient_status["regallDWIToT1wToT1wCommonSpace"] == True):
                try:
                    print("regallDWIToT1wToT1wCommonSpace",flush=True)
                    regallDWIToT1wToT1wCommonSpace(folder_path, p, DWI_type="AP", maskType=None, T1_filepath=None, T1wCommonSpace_filepath=T1_MNI, T1wCommonSpaceMask_filepath=T1_MNI_mask, metrics_dic={'_FA': 'dti', 'RD': 'dti', 'AD': 'dti', 'MD': 'dti'})
                    patient_status["regallDWIToT1wToT1wCommonSpace"] = True
                except Exception as e:
                    patient_status["regallDWIToT1wToT1wCommonSpace"] = False
                    error = True
                    ex_type, ex_value, ex_traceback = sys.exc_info()
                    printError(ex_type, ex_value, ex_traceback)
                    print(e, flush=True)
                with open(json_status_file,"w") as f:
                    json.dump(patient_status, f, indent = 6)
                    
        if not (patient_status.get('tracking') is not None and patient_status["tracking"] == True):
            try:
                print("tracking",flush=True)
                study.tracking(patient_list_m=[p])
                patient_status["tracking"] = True
            except Exception as e:
                patient_status["tracking"] = False
                error = True
                ex_type, ex_value, ex_traceback = sys.exc_info()
                printError(ex_type, ex_value, ex_traceback)
                print(e, flush=True)
            with open(json_status_file,"w") as f:
                json.dump(patient_status, f, indent = 6)
        
        if patient_status.get('regallDWIToT1wToT1wCommonSpace') is not None and patient_status["regallDWIToT1wToT1wCommonSpace"] == True and patient_status.get('tracking') is not None and patient_status["tracking"] == True:
            
            if not (patient_status.get('inverseTransformAtlas') is not None and patient_status["inverseTransformAtlas"] == True):
                try:
                    print("inverseTransformAtlas",flush=True)
                    atlas_path = "/CECI/proj/pilab/PermeableAccess/meditant_h67jkhFfG9uyD/" + "BN_Atlas_246_1mm_MNI.nii.gz"
                    inverseTransformAtlas(folder_path, p, atlasPath=atlas_path, atlasName="BN_246_1mm", DWI_type="AP")
                    patient_status["inverseTransformAtlas"] = True
                except Exception as e:
                    patient_status["inverseTransformAtlas"] = False
                    error = True
                    ex_type, ex_value, ex_traceback = sys.exc_info()
                    printError(ex_type, ex_value, ex_traceback)
                    print(e, flush=True)
                with open(json_status_file,"w") as f:
                    json.dump(patient_status, f, indent = 6)
            
            if not (patient_status.get('siftComputation') is not None and patient_status["siftComputation"] == True):
                try:
                    print("siftComputation",flush=True)
                    print("Starting SIFT")
                    fname="BN_246_1mm_InSubjectDWISpaceFrom_AP"
                    siftComputation(folder_path, p, msmtCSD=True)
                    patient_status["siftComputation"] = True
                except Exception as e:
                    patient_status["siftComputation"] = False
                    error = True
                    ex_type, ex_value, ex_traceback = sys.exc_info()
                    printError(ex_type, ex_value, ex_traceback)
                    print(e, flush=True)
                with open(json_status_file,"w") as f:
                    json.dump(patient_status, f, indent = 6)
        
            if patient_status.get('siftComputation') is not None and patient_status["siftComputation"] == True:
                if not (patient_status.get('connectivityMatrixSift') is not None and patient_status["connectivityMatrixSift"] == True):
                    try:
                        print("connectivityMatrixSift",flush=True)
                        fname="BN_246_1mm_InSubjectDWISpaceFrom_AP"
                        print("Starting conn matrix")
                        fname="BN_246_1mm_InSubjectDWISpaceFrom_AP"
                        connectivityMatrix(folder_path, p, label_fname=fname, input="SIFT")
                        patient_status["connectivityMatrixSift"] = True
                    except Exception as e:
                        patient_status["connectivityMatrixSift"] = False
                        error = True
                        ex_type, ex_value, ex_traceback = sys.exc_info()
                        printError(ex_type, ex_value, ex_traceback)
                        print(e, flush=True)
                    with open(json_status_file,"w") as f:
                        json.dump(patient_status, f, indent = 6)
                    
            if not (patient_status.get('connectivityMatrixTCKGEN') is not None and patient_status["connectivityMatrixTCKGEN"] == True):
                try:
                    print("connectivityMatrixTCKGEN",flush=True)
                    fname="BN_246_1mm_InSubjectDWISpaceFrom_AP"
                    print("Starting conn matrix")
                    fname="BN_246_1mm_InSubjectDWISpaceFrom_AP"
                    connectivityMatrix(folder_path, p, label_fname=fname, input="TCKGEN")
                    patient_status["connectivityMatrixTCKGEN"] = True
                except Exception as e:
                    patient_status["connectivityMatrixTCKGEN"] = False
                    error = True
                    ex_type, ex_value, ex_traceback = sys.exc_info()
                    printError(ex_type, ex_value, ex_traceback)
                    print(e, flush=True)
                with open(json_status_file,"w") as f:
                    json.dump(patient_status, f, indent = 6)
                    
                    
        if patient_status.get('wm_mask_AP') is not None and patient_status.get('noddi') is not None and patient_status["wm_mask_AP"] == True and patient_status["noddi"] == True:
            if not (patient_status.get('regallDWIToT1wToT1wCommonSpaceNoddi') is not None and patient_status["regallDWIToT1wToT1wCommonSpaceNoddi"] == True):
                try:
                    print("regallDWIToT1wToT1wCommonSpaceNoddi",flush=True)
                    regallDWIToT1wToT1wCommonSpace(folder_path, p, DWI_type="AP", maskType=None, T1_filepath=None, T1wCommonSpace_filepath=T1_MNI, T1wCommonSpaceMask_filepath=T1_MNI_mask, metrics_dic={'noddi_fiso': 'noddi', 'noddi_odi': 'noddi', 'noddi_icvf': 'noddi', 'noddi_fintra': 'noddi', 'noddi_fextra': 'noddi', 'noddi_fbundle': 'noddi'})
                    patient_status["regallDWIToT1wToT1wCommonSpaceNoddi"] = True
                except Exception as e:
                    patient_status["regallDWIToT1wToT1wCommonSpaceNoddi"] = False
                    error = True
                    ex_type, ex_value, ex_traceback = sys.exc_info()
                    printError(ex_type, ex_value, ex_traceback)
                    print(e, flush=True)
                with open(json_status_file,"w") as f:
                    json.dump(patient_status, f, indent = 6)
            
        if patient_status.get('wm_mask_AP') is not None and patient_status.get('fingerprinting') is not None and patient_status["wm_mask_AP"] == True and patient_status["fingerprinting"] == True:
            if not (patient_status.get('regallDWIToT1wToT1wCommonSpaceFingerprinting') is not None and patient_status["regallDWIToT1wToT1wCommonSpaceFingerprinting"] == True):
                try:
                    print("regallDWIToT1wToT1wCommonSpaceFingerprinting",flush=True)
                    regallDWIToT1wToT1wCommonSpace(folder_path, p, DWI_type="AP", maskType=None, T1_filepath=None, T1wCommonSpace_filepath=T1_MNI, T1wCommonSpaceMask_filepath=T1_MNI_mask, metrics_dic={'mf_fvf_f0': 'mf', 'mf_fvf_f1': 'mf', 'mf_fvf_tot': 'mf', 'mf_frac_f0': 'mf', 'mf_frac_f1': 'mf', 'mf_frac_csf': 'mf', 'mf_DIFF_ex_f0': 'mf', 'mf_DIFF_ex_f1': 'mf', 'mf_DIFF_ex_tot': 'mf'})
                    patient_status["regallDWIToT1wToT1wCommonSpaceFingerprinting"] = True
                except Exception as e:
                    patient_status["regallDWIToT1wToT1wCommonSpaceFingerprinting"] = False
                    error = True
                    ex_type, ex_value, ex_traceback = sys.exc_info()
                    printError(ex_type, ex_value, ex_traceback)
                    print(e, flush=True)
                with open(json_status_file,"w") as f:
                    json.dump(patient_status, f, indent = 6)
                    
        if patient_status.get('wm_mask_AP') is not None and patient_status.get('CHARMED_r3') is not None and patient_status["wm_mask_AP"] == True and patient_status["CHARMED_r3"] == True:
            if not (patient_status.get('regallDWIToT1wToT1wCommonSpaceCHARMED_r3') is not None and patient_status["regallDWIToT1wToT1wCommonSpaceCHARMED_r3"] == True):
                try:
                    print("regallDWIToT1wToT1wCommonSpaceCHARMED_r3",flush=True)
                    
                    regallDWIToT1wToT1wCommonSpace(folder_path, p, DWI_type="AP", maskType=None, T1_filepath=None, T1wCommonSpace_filepath=T1_MNI, T1wCommonSpaceMask_filepath=T1_MNI_mask,metrics_dic={'CHARMED_r3_FR': 'CHARMED_r3', 'CHARMED_r3_S0.s0': 'CHARMED_r3', 'CHARMED_r3_AIC': 'CHARMED_r3', 'CHARMED_r3_BIC': 'CHARMED_r3', 'CHARMED_r3_Tensor.theta': 'CHARMED_r3','CHARMED_r3_Tensor.d': 'CHARMED_r3', 'CHARMED_r3_Tensor.psi': 'CHARMED_r3', 'CHARMED_r3_Tensor.phi': 'CHARMED_r3', 'CHARMED_r3_Tensor.dperp1': 'CHARMED_r3', 'CHARMED_r3_Tensor.dperp0': 'CHARMED_r3','CHARMED_r3_Tensor.FA': 'CHARMED_r3', 'CHARMED_r3_Tensor.MD': 'CHARMED_r3', 'CHARMED_r3_Tensor.AD': 'CHARMED_r3', 'CHARMED_r3_Tensor.RD': 'CHARMED_r3','CHARMED_r3_w_': 'CHARMED_r3', 'CHARMED_r3_CHARMEDRestricted0.d': 'CHARMED_r3', 'CHARMED_r3_CHARMEDRestricted0.phi': 'CHARMED_r3', 'CHARMED_r3_CHARMEDRestricted0.theta': 'CHARMED_r3', 'CHARMED_r3_CHARMEDRestricted1.d': 'CHARMED_r3', 'CHARMED_r3_CHARMEDRestricted1.phi': 'CHARMED_r3', 'CHARMED_r3_CHARMEDRestricted1.theta': 'CHARMED_r3', 'CHARMED_r3_CHARMEDRestricted2.d': 'CHARMED_r3', 'CHARMED_r3_CHARMEDRestricted2.phi': 'CHARMED_r3', 'CHARMED_r3_CHARMEDRestricted2.theta': 'CHARMED_r3'})
                    patient_status["regallDWIToT1wToT1wCommonSpaceCHARMED_r3"] = True
                except Exception as e:
                    patient_status["regallDWIToT1wToT1wCommonSpaceCHARMED_r3"] = False
                    error = True
                    ex_type, ex_value, ex_traceback = sys.exc_info()
                    printError(ex_type, ex_value, ex_traceback)
                    print(e, flush=True)
                with open(json_status_file,"w") as f:
                    json.dump(patient_status, f, indent = 6)
                    
        with open(json_status_file,"w") as f:
            json.dump(patient_status, f, indent = 6)
            
        print(patient_status, flush=True)
        
    if error == True:
        raise Exception("One of the subscripts returned an error.")
        