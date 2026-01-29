import nibabel as nib
import matplotlib.pyplot as plt
import os
import glob

def visualize_sample():
    # 1. Path to dataset
    data_dir = os.path.join("data", "Task09_Spleen")
    images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    
    if not images:
        print("Error: No images found. Check the path to the dataset.")
        return
    
    if not labels:
        print("Error: No labels found. Check the path to the dataset.")
        return

    # 2. Load one sample (Image and Label)
    # Medical images are 3D (H, W, Depth)
    img_path = images[0]
    label_path = labels[0]

    img = nib.load(img_path).get_fdata()
    label = nib.load(label_path).get_fdata()
    
    print(f"Image shape: {img.shape}")
    print(f"Label shape: {label.shape}")

    # 3. Plot a slice (Z-axis, middle slice)
    slice_idx = img.shape[2] // 2
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title(f"CT Scan at Slice {slice_idx}")
    plt.imshow(img[:, :, slice_idx], cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Label (Spleen)")
    plt.imshow(label[:, :, slice_idx], cmap="gray")
    plt.axis("off")

    plt.show()
    
    # 4. Save the plot instead of showing it
    output_path = os.path.join("outputs", "sample_exploration.png")
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    visualize_sample()