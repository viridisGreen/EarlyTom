import os
from .utils import apply_tome
from .llava_arch import LlavaMetaForCausalLM_tome

def tome(model):
    
    print("################################")
    print("########## ToMe ###########")
    print("################################")
    
    r = int(os.environ.get("R", 0))
    merge_num = int(os.environ.get("MERGE_NUM", 0))
    
    apply_tome(model.model.vision_tower.vision_tower, r, merge_num)
    
    from llava.model.llava_arch import LlavaMetaForCausalLM
    LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = LlavaMetaForCausalLM_tome.prepare_inputs_labels_for_multimodal
    LlavaMetaForCausalLM.get_2dPool = LlavaMetaForCausalLM_tome.get_2dPool

    return model
