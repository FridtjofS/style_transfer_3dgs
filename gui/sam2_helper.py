import numpy as np

# from sam2 import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# point guided
def self_prompt(predictor, point_prompts, sam_feature):
    input_point = point_prompts.detach().cpu().numpy()
    # input_point = input_point[::-1]
    input_label = np.ones(len(input_point))

    predictor._features = sam_feature
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    # return_mask = (masks[ :, :, 0]*255).astype(np.uint8)
    # return_mask = (masks[id, :, :, None] * 255).astype(np.uint8)
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    # return highest scoring mask
    return_mask = (masks[0, :, :, None] * 255).astype(np.uint8)

    return return_mask / 255






