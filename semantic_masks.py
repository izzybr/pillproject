# some notes from semantic segretation using FCN50?

masks =  [
    normalized_masks[img_idx, sem_class_to_idx[cls]]
    for img_idx in range(len(imlist))
    for cls in enumerate(weights.meta["categories"])
]

masks = [
    draw_segmentation_masks(img, masks=mask, alpha=0.7)
    for img, mask in zip(imlist, boolean_dog_masks)
]