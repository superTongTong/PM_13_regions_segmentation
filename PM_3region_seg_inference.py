from Data_conversion.config import setup_nnunet
from pathlib import Path
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import os


def get_3_region_masks():
    device = 'cuda'
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
    setup_nnunet()
    nnUNet_raw = os.environ.get('nnUNet_raw')
    # nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
    nnUNet_results = os.environ.get('nnUNet_results')
    pretrained_path = Path('Dataset008_CKI_orig/finetune_500epochs2024_1_30_10_33')
    i_path = '/gpfs/work5/0/tesr0674/PM_13_regions_segmentation/data/pci_score_data/test_data/'
    o_path = Path('/gpfs/work5/0/tesr0674/PM_13_regions_segmentation/data/pci_score_data/masks_v2')
    # pretrained_path = Path('Dataset008_CKI_orig/finetune_500epochs2024_1_30_10_33')
    # i_path = './data/data_ViT/images/val/'
    # o_path = './data/data_ViT/masks_v2'
    pretrained_model_path = os.path.join(nnUNet_results, pretrained_path)
    # input_path = os.path.join(nnUNet_raw, i_path)
    # output_path = os.path.join(nnUNet_raw, o_path)
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(pretrained_model_path, use_folds=(0,),
                                                   checkpoint_name='checkpoint_best.pth')
    # variant 1: give input and output folders
    predictor.predict_from_files(i_path,
                                 o_path,
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=3, num_processes_segmentation_export=3,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)


def mian():

    get_3_region_masks()

if __name__ == "__main__":
    mian()
