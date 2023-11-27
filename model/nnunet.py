import os
import sys
import time
import subprocess
from pathlib import Path
import numpy as np
import nibabel as nib
from functools import partial
from p_tqdm import p_map
import tempfile
import torch
from utils.libs import nostdout

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.file_path_utilities import get_output_folder

from Data_conversion.map_to_binary import class_map_PM
from utils.alignment import as_closest_canonical, undo_canonical
from utils.resampling import change_spacing
from utils.libs import check_if_shape_and_affine_identical


def _get_full_task_name(task_id: int, src: str="raw"):
    if src == "raw":
        base = Path(os.environ['nnUNet_raw_data_base']) / "nnUNet_raw_data"
    elif src == "preprocessed":
        base = Path(os.environ['nnUNet_preprocessed'])
    elif src == "results":
        base = Path(os.environ['RESULTS_FOLDER']) / "nnUNet" / "3d_fullres"
    dirs = [str(dir).split("/")[-1] for dir in base.glob("*")]
    for dir in dirs:
        if f"Task{task_id:03d}" in dir:
            return dir

    # If not found in 3d_fullres, search in 3d_lowres
    if src == "results":
        base = Path(os.environ['RESULTS_FOLDER']) / "nnUNet" / "3d_lowres"
        dirs = [str(dir).split("/")[-1] for dir in base.glob("*")]
        for dir in dirs:
            if f"Task{task_id:03d}" in dir:
                return dir

    # If not found in 3d_lowres, search in 2d
    if src == "results":
        base = Path(os.environ['RESULTS_FOLDER']) / "nnUNet" / "2d"
        dirs = [str(dir).split("/")[-1] for dir in base.glob("*")]
        for dir in dirs:
            if f"Task{task_id:03d}" in dir:
                return dir

    raise ValueError(f"task_id {task_id} not found")


def contains_empty_img(imgs):
    """
    imgs: List of image pathes
    """
    is_empty = True
    for img in imgs:
        this_is_empty = len(np.unique(nib.load(img).get_fdata())) == 1
        is_empty = is_empty and this_is_empty
    return is_empty


def nnUNetv2_predict(dir_in, dir_out, task_id, model="3d_fullres", folds=None,
                     trainer="nnUNetTrainer", tta=False,
                     num_threads_preprocessing=3, num_threads_nifti_save=2,
                     plans="nnUNetPlans", device="cuda", quiet=False):
    """
    Identical to bash function nnUNetv2_predict

    folds:  folds to use for prediction. Default is None which means that folds will be detected 
            automatically in the model output folder.
            for all folds: None
            for only fold 0: [0]
    """
    dir_in = str(dir_in)
    dir_out = str(dir_out)

    model_folder = get_output_folder(task_id, trainer, plans, model)

    assert device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {device}.'
    if device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        # torch.set_num_interop_threads(1)  # throws error if setting the second time
        device = torch.device('cuda')
    else:
        device = torch.device('mps')
    step_size = 0.5
    disable_tta = not tta
    verbose = False
    save_probabilities = False
    continue_prediction = False
    chk = "checkpoint_final.pth"
    npp = num_threads_preprocessing
    nps = num_threads_nifti_save
    prev_stage_predictions = None
    num_parts = 1
    part_id = 0
    allow_tqdm = not quiet

    # predict_from_raw_data(dir_in,
    #                       dir_out,
    #                       model_folder,
    #                       folds,
    #                       step_size,
    #                       use_gaussian=True,
    #                       use_mirroring=not disable_tta,
    #                       perform_everything_on_gpu=True,
    #                       verbose=verbose,
    #                       save_probabilities=save_probabilities,
    #                       overwrite=not continue_prediction,
    #                       checkpoint_name=chk,
    #                       num_processes_preprocessing=npp,
    #                       num_processes_segmentation_export=nps,
    #                       folder_with_segs_from_prev_stage=prev_stage_predictions,
    #                       num_parts=num_parts,
    #                       part_id=part_id,
    #                       device=device)

    predictor = nnUNetPredictor(
        tile_step_size=step_size,
        use_gaussian=True,
        use_mirroring=not disable_tta,
        perform_everything_on_gpu=True,
        device=device,
        verbose=verbose,
        verbose_preprocessing=verbose,
        allow_tqdm=allow_tqdm
    )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=folds,
        checkpoint_name=chk,
    )
    predictor.predict_from_files(dir_in, dir_out,
                                 save_probabilities=save_probabilities, overwrite=not continue_prediction,
                                 num_processes_preprocessing=npp, num_processes_segmentation_export=nps,
                                 folder_with_segs_from_prev_stage=prev_stage_predictions, 
                                 num_parts=num_parts, part_id=part_id)


def save_segmentation_nifti(class_map_item, tmp_dir=None, file_out=None, nora_tag=None, header=None, task_name=None, quiet=None):
    k, v = class_map_item
    # Have to load img inside of each thread. If passing it as argument a lot slower.
    if not task_name.startswith("total") and not quiet:
        print(f"Creating {v}.nii.gz")
    img = nib.load(tmp_dir / "s01.nii.gz")
    img_data = img.get_fdata()
    binary_img = img_data == k
    output_path = str(file_out / f"{v}.nii.gz")
    nib.save(nib.Nifti1Image(binary_img.astype(np.uint8), img.affine, header), output_path)
    if nora_tag != "None":
        subprocess.call(f"/opt/nora/src/node/nora -p {nora_tag} --add {output_path} --addtag mask", shell=True)


def nnUNet_predict_image(file_in, file_out, task_id, model="3d_fullres", folds=None,
                         trainer="nnUNetTrainerV2", tta=False,
                         resample=None, task_name="total", nora_tag="None",
                         nr_threads_resampling=1, nr_threads_saving=6, force_split=False,
                         roi_subset=None, output_type="nifti",
                         quiet=False, verbose=False, test=0, skip_saving=False,
                         device="cuda"):
    """
    crop: string or a nibabel image
    resample: None or float  (target spacing for all dimensions)
    """
    file_in = Path(file_in)
    if file_out is not None:
        file_out = Path(file_out)
    if not file_in.exists():
        sys.exit("ERROR: The input file or directory does not exist.")
    multimodel = type(task_id) is list

    img_type = "nifti" if str(file_in).endswith(".nii") or str(file_in).endswith(".nii.gz") else "dicom"

    if img_type == "nifti" and output_type == "dicom":
        raise ValueError("To use output type dicom you also have to use a Dicom image as input.")

    # for debugging
    # tmp_dir = file_in.parent / ("nnunet_tmp_" + ''.join(random.Random().choices(string.ascii_uppercase + string.digits, k=8)))
    # (tmp_dir).mkdir(exist_ok=True)
    # with tmp_dir as tmp_folder:
    with tempfile.TemporaryDirectory(prefix="nnunet_tmp_") as tmp_folder:
        tmp_dir = Path(tmp_folder)

        img_in_orig = nib.load(file_in)
        if len(img_in_orig.shape) == 2:
            raise ValueError("TotalSegmentator does not work for 2D images. Use a 3D image.")
        if len(img_in_orig.shape) > 3:
            print(f"WARNING: Input image has {len(img_in_orig.shape)} dimensions. Only using first three dimensions.")
            img_in_orig = nib.Nifti1Image(img_in_orig.get_fdata()[:,:,:,0], img_in_orig.affine)
        
        # takes ~0.9s for medium image
        img_in = nib.Nifti1Image(img_in_orig.get_fdata(), img_in_orig.affine)  # copy img_in_orig

        img_in = as_closest_canonical(img_in)

        if resample is not None:
            if not quiet: print(f"Resampling...")
            st = time.time()
            img_in_shape = img_in.shape
            # img_in_zooms = img_in.header.get_zooms()
            img_in_rsp = change_spacing(img_in, [resample, resample, resample],
                                        order=3, dtype=np.int32, nr_cpus=nr_threads_resampling)  # 4 cpus instead of 1 makes it a bit slower
            if verbose:
                print(f"  from shape {img_in.shape} to shape {img_in_rsp.shape}")
            if not quiet: print(f"  Resampled in {time.time() - st:.2f}s")
        else:
            img_in_rsp = img_in

        nib.save(img_in_rsp, tmp_dir / "s01_0000.nii.gz")

        # nr_voxels_thr = 512*512*900
        nr_voxels_thr = 256*256*900
        img_parts = ["s01"]
        ss = img_in_rsp.shape
        # If image to big then split into 3 parts along z axis. Also make sure that z-axis is at least 200px otherwise
        # splitting along it does not really make sense.
        do_triple_split = np.prod(ss) > nr_voxels_thr and ss[2] > 200 and multimodel
        if force_split:
            do_triple_split = True
        if do_triple_split:
            if not quiet: print(f"Splitting into subparts...")
            img_parts = ["s01", "s02", "s03"]
            third = img_in_rsp.shape[2] // 3
            margin = 20  # set margin with fixed values to avoid rounding problem if using percentage of third
            img_in_rsp_data = img_in_rsp.get_fdata()
            nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, :third+margin], img_in_rsp.affine),
                    tmp_dir / "s01_0000.nii.gz")
            nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, third+1-margin:third*2+margin], img_in_rsp.affine),
                    tmp_dir / "s02_0000.nii.gz")
            nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, third*2+1-margin:], img_in_rsp.affine),
                    tmp_dir / "s03_0000.nii.gz")

        if not quiet: print(f"Predicting...")
        if test == 0:
            with nostdout(verbose):
                nnUNetv2_predict(tmp_dir, tmp_dir, task_id, model, folds, trainer, tta,
                                 nr_threads_resampling, nr_threads_saving, device=device, quiet=quiet)

        if not quiet: print("  Predicted in {:.2f}s".format(time.time() - st))

        # Combine image subparts back to one image
        if do_triple_split:
            combined_img = np.zeros(img_in_rsp.shape, dtype=np.uint8)
            combined_img[:,:,:third] = nib.load(tmp_dir / "s01.nii.gz").get_fdata()[:,:,:-margin]
            combined_img[:,:,third:third*2] = nib.load(tmp_dir / "s02.nii.gz").get_fdata()[:,:,margin-1:-margin]
            combined_img[:,:,third*2:] = nib.load(tmp_dir / "s03.nii.gz").get_fdata()[:,:,margin-1:]
            nib.save(nib.Nifti1Image(combined_img, img_in_rsp.affine), tmp_dir / "s01.nii.gz")

        img_pred = nib.load(tmp_dir / "s01.nii.gz")

        if resample is not None:
            if not quiet: print("Resampling...")
            if verbose: print(f"  back to original shape: {img_in_shape}")    
            # Use force_affine otherwise output affine sometimes slightly off (which then is even increased
            # by undo_canonical)
            img_pred = change_spacing(img_pred, [resample, resample, resample], img_in_shape,
                                        order=0, dtype=np.uint8, nr_cpus=nr_threads_resampling, 
                                        force_affine=img_in.affine)

        img_pred = undo_canonical(img_pred, img_in_orig)

        check_if_shape_and_affine_identical(img_in_orig, img_pred)

        img_data = img_pred.get_fdata().astype(np.uint8)

        if file_out is not None and skip_saving is False:
            if not quiet: print("Saving segmentations...")

            new_header = img_in_orig.header.copy()
            new_header.set_data_dtype(np.uint8)

            st = time.time()
            selected_classes = class_map_organs[task_name]
            file_out.mkdir(exist_ok=True, parents=True)
            nib.save(img_pred, tmp_dir / "s01.nii.gz")
            _ = p_map(partial(save_segmentation_nifti, tmp_dir=tmp_dir, file_out=file_out, nora_tag=nora_tag, header=new_header, task_name=task_name, quiet=quiet),
                        selected_classes.items(), num_cpus=nr_threads_saving, disable=quiet)

            if not quiet: print(f"  Saved in {time.time() - st:.2f}s")

    return nib.Nifti1Image(img_data, img_pred.affine)
