"""Post-processing: resample prediction to original image grid."""

import nibabel as nib
from monai.transforms import ResampleToMatchd, Compose, LoadImaged, EnsureChannelFirstd


def resample_to_original(image_orig_path, pred_path, output_path):
    """Resample the prediction to match the original image dimensions and header."""
    data = {
        "image_orig": image_orig_path,
        "pred": pred_path,
    }
    loader = Compose([
        LoadImaged(keys=["image_orig", "pred"]),
        EnsureChannelFirstd(keys=["image_orig", "pred"]),
    ])
    loaded_data = loader(data)

    resampler = ResampleToMatchd(
        keys=["pred"],
        key_dst="image_orig",
        mode="nearest",
    )
    final_data = resampler(loaded_data)

    orig_img = nib.load(image_orig_path)
    pred_array = final_data["pred"][0].detach().cpu().numpy()
    new_seg = nib.Nifti1Image(pred_array, orig_img.affine, orig_img.header)
    nib.save(new_seg, output_path)
    print(f"Resampled prediction saved to: {output_path}")
