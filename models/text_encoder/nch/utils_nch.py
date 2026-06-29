import os
import torch


class TrainableVector_multitask():
    def __init__(self, 
                 vlm_text_encoder=None,
                 t2i_fix_tokens=False, 
                 bbox_fix_tokens=False, 
                 ctxt_fix_tokens=False, 
                 seg_fix_tokens=False, 
                 sr_fix_tokens=False, 
                 sun_fix_tokens=False, 
                 facesr_fix_tokens=False, 
                 sr20Xbright_fix_tokens=False, 
                 sr20Xdark_fix_tokens=False, 
                 sr50X_fix_tokens=False, 
                 bokeh_fix_tokens=False, 
                 linedraw_fix_tokens=False,
                 moonsr_fix_tokens=False,
                 moonpainting_fix_tokens=False,
                 refsr_fix_tokens=False,
                 dehalo_fix_tokens=False,
                 firework_fix_tokens=False,
                 token_requires_grad=True, 
                 token_len=10):
        self.tokenizer = vlm_text_encoder.tokenizer
        self.text_encoder = vlm_text_encoder.model

        self.ori_embeds_params = None

        self.t2i_fix_tokens = ['p_t2i_'+str(i) for i in range(token_len)] if t2i_fix_tokens else []
        self.bbox_fix_tokens = ['p_bbox_'+str(i) for i in range(token_len)] if bbox_fix_tokens else []
        self.ctxt_fix_tokens = ['p_ctxt_'+str(i) for i in range(token_len)] if ctxt_fix_tokens else []
        self.seg_fix_tokens = ['p_seg_'+str(i) for i in range(token_len)] if seg_fix_tokens else []
        self.sr_fix_tokens = ['p_sr_'+str(i) for i in range(token_len)] if sr_fix_tokens else []
        self.sun_fix_tokens = ['p_sun_'+str(i) for i in range(token_len)] if sun_fix_tokens else []
        self.facesr_fix_tokens = ['p_facesr_'+str(i) for i in range(token_len)] if facesr_fix_tokens else []
        self.sr20Xbright_fix_tokens = ['p_sr20Xbright_'+str(i) for i in range(token_len)] if sr20Xbright_fix_tokens else []
        self.sr20Xdark_fix_tokens = ['p_sr20Xdark_'+str(i) for i in range(token_len)] if sr20Xdark_fix_tokens else []
        self.sr50X_fix_tokens = ['p_sr50X_'+str(i) for i in range(token_len)] if sr50X_fix_tokens else []
        self.bokeh_fix_tokens = ['p_bokeh_'+str(i) for i in range(token_len)] if bokeh_fix_tokens else []
        self.linedraw_fix_tokens = ['p_linedraw_'+str(i) for i in range(token_len)] if linedraw_fix_tokens else []
        self.moonsr_fix_tokens = ['p_moonsr_'+str(i) for i in range(token_len)] if moonsr_fix_tokens else []
        self.moonpainting_fix_tokens = ['p_moonpainting_'+str(i) for i in range(token_len)] if moonpainting_fix_tokens else []
        self.refsr_fix_tokens = ['p_refsr_'+str(i) for i in range(token_len)] if refsr_fix_tokens else []
        self.dehalo_fix_tokens = ['p_dehalo_'+str(i) for i in range(token_len)] if dehalo_fix_tokens else []
        self.firework_fix_tokens = ['p_firework_'+str(i) for i in range(token_len)] if firework_fix_tokens else []

        self.task_tokens = {'inpainting_bbox': ','.join(self.bbox_fix_tokens)+'. ' if self.bbox_fix_tokens else '',
                            'inpainting_ctxt':','.join(self.ctxt_fix_tokens)+'. ' if self.ctxt_fix_tokens else '',
                            'inpainting_seg':','.join(self.seg_fix_tokens)+'. ' if self.seg_fix_tokens else '',
                            'sr':','.join(self.sr_fix_tokens)+'. ' if self.sr_fix_tokens else '',
                            't2i':','.join(self.t2i_fix_tokens)+'. ' if self.t2i_fix_tokens else '',
                            'sun':','.join(self.sun_fix_tokens)+'. ' if self.sun_fix_tokens else '',
                            'facesr':','.join(self.facesr_fix_tokens)+'. ' if self.facesr_fix_tokens else '',
                            'sr20Xbright':','.join(self.sr20Xbright_fix_tokens)+'. ' if self.sr20Xbright_fix_tokens else '',
                            'sr20Xdark':','.join(self.sr20Xdark_fix_tokens)+'. ' if self.sr20Xdark_fix_tokens else '',
                            'sr50X':','.join(self.sr50X_fix_tokens)+'. ' if self.sr50X_fix_tokens else '',
                            'bokeh':','.join(self.bokeh_fix_tokens)+'. ' if self.bokeh_fix_tokens else '',
                            'linedraw':','.join(self.linedraw_fix_tokens)+'. ' if self.linedraw_fix_tokens else '',
                            'moonsr':','.join(self.moonsr_fix_tokens)+'. ' if self.moonsr_fix_tokens else '',
                            'moonpainting':','.join(self.moonpainting_fix_tokens)+'. ' if self.moonpainting_fix_tokens else '',
                            'refsr':','.join(self.refsr_fix_tokens)+'. ' if self.refsr_fix_tokens else '',
                            'dehalo':','.join(self.dehalo_fix_tokens)+'. ' if self.dehalo_fix_tokens else '',
                            'firework':','.join(self.firework_fix_tokens)+'. ' if self.firework_fix_tokens else '',
                            }
        
        self.custom_tokens = (
            self.t2i_fix_tokens
            + self.bbox_fix_tokens
            + self.ctxt_fix_tokens
            + self.seg_fix_tokens
            + self.sr_fix_tokens
            + self.sun_fix_tokens
            + self.facesr_fix_tokens
            + self.sr20Xbright_fix_tokens
            + self.sr20Xdark_fix_tokens
            + self.sr50X_fix_tokens
            + self.bokeh_fix_tokens
            + self.linedraw_fix_tokens
            + self.moonsr_fix_tokens
            + self.moonpainting_fix_tokens
            + self.refsr_fix_tokens
            + self.dehalo_fix_tokens
            + self.firework_fix_tokens
        )        

        self.with_fix_token = (
            self.t2i_fix_tokens
            or self.bbox_fix_tokens
            or self.ctxt_fix_tokens
            or self.seg_fix_tokens
            or self.sr_fix_tokens
            or self.sun_fix_tokens
            or self.facesr_fix_tokens
            or self.sr20Xbright_fix_tokens
            or self.sr20Xdark_fix_tokens
            or self.sr50X_fix_tokens
            or self.bokeh_fix_tokens
            or self.linedraw_fix_tokens
            or self.moonsr_fix_tokens
            or self.moonpainting_fix_tokens     
            or self.refsr_fix_tokens      
            or self.dehalo_fix_tokens   
            or self.firework_fix_tokens          
        )
        
        self.token_len = token_len # 暂不支持10以外
        self.token_requires_grad = token_requires_grad
        
        self.task_initializer_prompts = {
            't2i': "Please generate a high quality image with the following prompt",
            'bbox': "Please inpaint bbox mask area precisely with the following prompt",
            'ctxt': "Please fill the mask area naturally according to the background",
            'seg': "Please inpaint segment mask area precisely with the following prompt",
            'sr': "Please super resolve image to high quality and sharp details",
            'sun': "Please restore the shape of sun with the following prompt",
            'facesr': "Please enhance the face to high quality and high resolution",
            'sr20Xbright': "Super resolve image of 20X bright scene to high quality",
            'sr20Xdark': "Super resolve image of 20X dark scene to high quality",
            'sr50X': "Please super resolve image of 50X scene to high quality", 
            'bokeh': "Please generate accurate and natural bokeh effect for the image",
            'linedraw': "Please extract 3D full-body pose lines with the following prompt",
            'moonsr': "Please super resolve moon to high quality and sharp details",
            'moonpainting': "Please generate a high quality moon with the following prompt",
            'refsr': "Please super resolve image to high quality and sharp details",
            'dehalo': "Please remove the halo effect and keep details of image",
            'firework': "Please turn the fireworks into continuous streaks of the image",
        }

    def _get_prompt_embeddings(self, prompt):
        """辅助函数：获取prompt的10个token的embedding列表（一对一初始化用）"""
        # 分词并转换为token id（确保10个，已在校验中保证）
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        # 获取embedding层
        embed_layer = self.text_encoder.language_model.get_input_embeddings()
        # 逐个获取每个token的embedding，返回列表（长度10）
        prompt_embeds = [embed_layer.weight[token_id].clone() for token_id in prompt_ids]
        
        return prompt_embeds
    
    def add_new_token(self):
        if self.with_fix_token:

            # original_vocab_size = len(self.tokenizer)

            # 2026.02.11 添加自定义 token
            self.tokenizer.add_tokens(self.custom_tokens)
            self.text_encoder.language_model.resize_token_embeddings(len(self.tokenizer))
            self.text_encoder.config.llm_config.vocab_size = len(self.tokenizer)
            self.text_encoder.language_model.config.vocab_size = len(self.tokenizer)


            token_embeds = self.text_encoder.language_model.get_input_embeddings().weight.data.clone()
            with torch.no_grad():
                # 遍历每个任务，初始化对应token
                task_token_mapping = {
                    't2i': self.t2i_fix_tokens,
                    'bbox': self.bbox_fix_tokens,
                    'ctxt': self.ctxt_fix_tokens,
                    'seg': self.seg_fix_tokens,
                    'sr': self.sr_fix_tokens,
                    'sun': self.sun_fix_tokens,
                    'facesr': self.facesr_fix_tokens,
                    'sr20Xbright': self.sr20Xbright_fix_tokens,
                    'sr20Xdark': self.sr20Xdark_fix_tokens,
                    'sr50X': self.sr50X_fix_tokens,
                    'bokeh': self.bokeh_fix_tokens,
                    'linedraw': self.linedraw_fix_tokens,
                    'moonsr': self.moonsr_fix_tokens,
                    'moonpainting': self.moonpainting_fix_tokens,
                    'refsr': self.refsr_fix_tokens,
                    'dehalo': self.dehalo_fix_tokens,
                    'firework': self.firework_fix_tokens,
                }
                
                for task, tokens in task_token_mapping.items():
                    if not tokens:
                        continue  # 该任务未启用自定义token，跳过
                    
                    # 校验：自定义token数必须是10个
                    if len(tokens) != self.token_len:
                        raise ValueError(f"任务[{task}]的自定义token数为{len(tokens)}，要求必须是{self.token_len}个！")
                    
                    prompt_embeds = self._get_prompt_embeddings(self.task_initializer_prompts[task])
                    
                    for i, token in enumerate(tokens):
                        token_id = self.tokenizer.convert_tokens_to_ids(token)
                        token_embeds[token_id] = prompt_embeds[i].clone()


            # 创建新的 Embedding 层
            new_embeddings = torch.nn.Embedding.from_pretrained(
                token_embeds,
                freeze=True
            )
            
            new_embeddings.weight.requires_grad = self.token_requires_grad
            # 替换模型中的 embedding 层
            self.text_encoder.language_model.set_input_embeddings(new_embeddings)
            self.ori_embeds_params = self.text_encoder.language_model.get_input_embeddings().weight.data.clone()

    def reset_old_token(self):
        if self.with_fix_token and self.token_requires_grad:

            index_no_updates = torch.ones((len(self.tokenizer),), dtype=torch.bool)
            new_token_ids = self.tokenizer.convert_tokens_to_ids(self.custom_tokens)
            min_token_id = int(min(new_token_ids))
            max_token_id = int(max(new_token_ids))
            index_no_updates[min_token_id : max_token_id + 1] = False
            
            with torch.no_grad():
                self.text_encoder.language_model.get_input_embeddings().weight[index_no_updates] = self.ori_embeds_params[index_no_updates]


    def load_model(self, path, logger, mp_rank, mp_world_size):
        """
        加载VLM模型的embedding权重（兼容多进程和DDP）
        参数说明：
            path: 权重保存目录
            logger: 日志对象（可为None）
            mp_rank: 当前进程rank
            mp_world_size: 总进程数
        """
        if self.with_fix_token:
            # 构造权重文件名（保持和T5/CLIP一致的命名风格）
            weight_filename = (
                f"consolidated_vlm_llm.{mp_rank:02d}-of-{mp_world_size:02d}.pth"
            )
            weight_path = os.path.join(path, weight_filename)
            
            if logger:
                logger.info(f"Resuming VLM_LLM weights from: {weight_path}")
            
            # 加载权重（先加载到CPU避免设备不匹配）
            state_dict = torch.load(weight_path, map_location="cpu")
            
            # 获取真实的language_model（处理DDP包装的情况）
            language_model = self.text_encoder.language_model
            if hasattr(language_model, 'module') and isinstance(language_model.module, torch.nn.Module):
                language_model = language_model.module
            
            # 严格加载权重（确保embedding层参数完全匹配）
            language_model.load_state_dict(state_dict, strict=True)
            
            # 重新赋值回原对象（保持引用一致）
            if hasattr(self.text_encoder.language_model, 'module'):
                self.text_encoder.language_model.module.load_state_dict(state_dict, strict=True)
            else:
                self.text_encoder.language_model.load_state_dict(state_dict, strict=True)
            
            # 刷新ori_embeds_params（确保reset_old_token逻辑正常）
            self.ori_embeds_params = self.text_encoder.language_model.get_input_embeddings().weight.data.clone()
            
            if logger:
                logger.info(f"Successfully loaded VLM_LLM weights for rank {mp_rank}")

    def save_model(self, path, mp_rank, mp_world_size):
        """
        保存VLM模型的embedding权重（兼容多进程和DDP）
        参数说明：
            path: 权重保存目录
            mp_rank: 当前进程rank
            mp_world_size: 总进程数
        """
        if self.with_fix_token:
            # 创建保存目录（如果不存在）
            os.makedirs(path, exist_ok=True)
            
            # 获取真实的language_model（处理DDP包装的情况）
            language_model = self.text_encoder.language_model
            if hasattr(language_model, 'module') and isinstance(language_model.module, torch.nn.Module):
                state_dict = language_model.module.state_dict()
            else:
                state_dict = language_model.state_dict()
            
            # 构造权重文件名（保持和T5/CLIP一致的命名风格）
            weight_filename = (
                f"consolidated_vlm_llm.{mp_rank:02d}-of-{mp_world_size:02d}.pth"
            )
            weight_path = os.path.join(path, weight_filename)
            
            # 保存权重
            torch.save(state_dict, weight_path)
            
            # 打印日志（可选）
            print(f"Successfully saved VLM_LLM weights to {weight_path}")
