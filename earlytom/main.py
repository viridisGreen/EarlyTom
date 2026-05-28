import os
from .llava_arch import LlavaMetaForCausalLM_earlytom
from .modeling_qwen2 import Qwen2Model_earlytom

def earlytom(model):
    
    print("################################")
    print("############ EarlyTom ###########")
    print("################################")

    from llava.model.llava_arch import LlavaMetaForCausalLM
    LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = LlavaMetaForCausalLM_earlytom.prepare_inputs_labels_for_multimodal
    LlavaMetaForCausalLM.encode_images = LlavaMetaForCausalLM_earlytom.encode_images
    LlavaMetaForCausalLM.encode_images_multi = LlavaMetaForCausalLM_earlytom.encode_images_multi
    
    LlavaMetaForCausalLM.cluster_dpc_knn = LlavaMetaForCausalLM_earlytom.cluster_dpc_knn
    LlavaMetaForCausalLM.merge_tokens_by_clustering = LlavaMetaForCausalLM_earlytom.merge_tokens_by_clustering
    LlavaMetaForCausalLM.add_newline_token = LlavaMetaForCausalLM_earlytom.add_newline_token

    LlavaMetaForCausalLM.divided_static_dynamic = LlavaMetaForCausalLM_earlytom.divided_static_dynamic
    LlavaMetaForCausalLM.spatial_compression = LlavaMetaForCausalLM_earlytom.spatial_compression
    LlavaMetaForCausalLM.merge_tokens_by_attention_frame = LlavaMetaForCausalLM_earlytom.merge_tokens_by_attention_frame
    LlavaMetaForCausalLM.merge_tokens_by_local_window = LlavaMetaForCausalLM_earlytom.merge_tokens_by_local_window

    
    if os.getenv("INNER_k") is not None and os.getenv("INNER_r") is not None:
        print("INNER")
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
        Qwen2Model.forward = Qwen2Model_earlytom.forward
    else:
        print("INNER (w/o M)")
    
    return model
