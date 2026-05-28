import os
from .llava_arch import LlavaMetaForCausalLM_fastvid, LlavaMetaModel_fastvid
from .modeling_qwen2 import Qwen2Model_fastvid

def fastvid(model):
    
    print("################################")
    print("############ FastVid ###########")
    print("################################")

    print("FastVid")
    # from llava.model.llava_arch import LlavaMetaModel
    # LlavaMetaModel.build_vision_abstract = LlavaMetaModel_fastvid.build_vision_abstract
    # LlavaMetaModel.get_vision_abstract = LlavaMetaModel_fastvid.get_vision_abstract
    # LlavaMetaModel.__init__ = LlavaMetaModel_fastvid.__init__

    from llava.model.llava_arch import LlavaMetaForCausalLM
    LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = LlavaMetaForCausalLM_fastvid.prepare_inputs_labels_for_multimodal
    LlavaMetaForCausalLM.build_vision_abstract = LlavaMetaForCausalLM_fastvid.build_vision_abstract
    LlavaMetaForCausalLM.get_vision_abstract = LlavaMetaForCausalLM_fastvid.get_vision_abstract
    LlavaMetaForCausalLM.encode_images = LlavaMetaForCausalLM_fastvid.encode_images
    LlavaMetaForCausalLM.get_attn_2dPool = LlavaMetaForCausalLM_fastvid.get_attn_2dPool
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
    Qwen2Model.forward = Qwen2Model_fastvid.forward
    Qwen2Model.set_my_kwargs = Qwen2Model_fastvid.set_my_kwargs
    
    return model