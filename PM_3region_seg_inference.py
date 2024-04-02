from Data_conversion.config import setup_nnunet
from pathlib import Path
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import os


def get_3_region_masks():

    setup_nnunet()
    nnUNet_raw = os.environ.get('nnUNet_raw')
    # nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
    nnUNet_results = os.environ.get('nnUNet_results')
    pretrained_path = Path('Dataset017_Resampled_nn_scratch/nnUNetTrainer_500epochs2024_3_23_13_55')
    i_path = Path('Dataset017_Resampled_nn_scratch/imagesTs')
    o_path = Path('Dataset017_Resampled_nn_scratch/imagesTs_pred')
    pretrained_model_path = os.path.join(nnUNet_results, pretrained_path)
    input_path = os.path.join(nnUNet_raw, i_path)
    output_path = os.path.join(nnUNet_raw, o_path)
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(pretrained_model_path, use_folds=(0,),
                                                   checkpoint_name='checkpoint_best.pth')
    # variant 1: give input and output folders
    predictor.predict_from_files(input_path,
                                 output_path,
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)


def mian():

    get_3_region_masks()

if __name__ == "__main__":
    mian()
