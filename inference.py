from Data_conversion.config import setup_nnunet
from pathlib import Path
import torch


def totalsegmentator(input, output, nr_thr_resamp=1, nr_thr_saving=6,
                     nora_tag="None", task="total", roi_subset=None,
                     force_split=False, output_type="nifti",
                     skip_saving=False, device="gpu",):

    input = Path(input)
    output = Path(output)

    nora_tag = "None" if nora_tag is None else nora_tag

    # available devices: gpu | cpu | mps
    if device == "gpu": device = "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        print(
            "No GPU detected. Running on CPU. This can be very slow. "
            "The '--fast' or the `--roi_subset` option can help to reduce runtime.")
        device = "cpu"

    setup_nnunet()

    from model.nnunet import nnUNet_predict_image  # this has to be after setting new env vars

    crop_addon = [3, 3, 3]  # default value
    task_id = 291
    resample = 1.5
    trainer = "nnUNetTrainerNoMirroring"
    crop = None
    model = "3d_fullres"

    if roi_subset is not None and type(roi_subset) is not list:
        raise ValueError("roi_subset must be a list of strings")
    if roi_subset is not None and task != "total":
        raise ValueError("roi_subset only works with task 'total'")

    folds = [0]  # None
    seg_img = nnUNet_predict_image(input, output, task_id, model=model, folds=folds,
                                   trainer=trainer, tta=False, resample=resample,
                                   task_name=task, nora_tag=nora_tag,
                                   nr_threads_resampling=nr_thr_resamp, nr_threads_saving=nr_thr_saving,
                                   force_split=force_split, roi_subset=roi_subset,
                                   output_type=output_type,
                                   skip_saving=skip_saving, device=device)

    return seg_img


def mian():

    input_folder = "../data/Case_00001_0000.nii.gz"
    output_folder = "../seg_output"
    totalsegmentator(input_folder, output_folder, nr_thr_resamp=1, nr_thr_saving=6,
                     nora_tag="None", task="class_map_part_organs", roi_subset=None,
                     force_split=False, output_type="nifti",
                     skip_saving=False, device="gpu")


if __name__ == "__main__":
    mian()
