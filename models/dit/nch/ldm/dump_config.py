import os
from pathlib import Path
from typing import Optional

DUMP_OFFLINE_INPUT_PATH = None
DUMP_INIT_DATA_PATH = None
DUMP_PER_BLOCK_RESULT_PATH = None
DUMP_PATH = None
DUMP_LINEAR_PATH = None
INPUT_SHAPE = None


class DumpConfig:
    _instance: Optional['DumpConfig'] = None
    
    def __init__(self):
        self._enable = os.getenv("DUMP_ENABLE", "false").lower() in ("true", "1", "yes")
        self._dump_input = os.getenv("DUMP_INPUT", "false").lower() in ("true", "1", "yes")
        self._dump_init = os.getenv("DUMP_INIT", "false").lower() in ("true", "1", "yes")
        self._dump_blocks = os.getenv("DUMP_BLOCKS", "false").lower() in ("true", "1", "yes")
        self._dump_attn = os.getenv("DUMP_ATTN", "false").lower() in ("true", "1", "yes")
        self._dump_linear = os.getenv("DUMP_LINEAR", "false").lower() in ("true", "1", "yes")
        
        self._base_path = os.getenv("DUMP_BASE_PATH", "dump")
        
        self._input_shape = os.getenv("INPUT_SHAPE", None)
        
        self._current_step = 1
    
    @classmethod
    def get_instance(cls) -> 'DumpConfig':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls):
        cls._instance = None
    
    @property
    def current_step(self) -> int:
        return self._current_step
    
    @current_step.setter
    def current_step(self, value: int):
        if value != self._current_step:
            self._current_step = value
            self._update_global_paths()
            self._reset_counters()
    
    def _reset_counters(self):
        from . import transformer_nch_v3_split as transformer_nch
        from . import normalization
        transformer_nch.double_block_cnt = 0
        transformer_nch.cnt_forbbit = 0
        normalization.offline_input_cnt = 0
    
    def _update_global_paths(self):
        global DUMP_OFFLINE_INPUT_PATH, DUMP_INIT_DATA_PATH, DUMP_PER_BLOCK_RESULT_PATH, DUMP_LINEAR_PATH
        DUMP_OFFLINE_INPUT_PATH = self.offline_input_path
        DUMP_INIT_DATA_PATH = self.init_data_path
        DUMP_PER_BLOCK_RESULT_PATH = self.per_block_result_path
        DUMP_LINEAR_PATH = self.linear_path
    
    @property
    def enable(self) -> bool:
        return self._enable and any([
            self._dump_input, self._dump_init, self._dump_blocks, self._dump_attn, self._dump_linear
        ])
    
    @property
    def dump_input(self) -> bool:
        return self.enable and self._dump_input
    
    @property
    def dump_init(self) -> bool:
        return self.enable and self._dump_init
    
    @property
    def dump_blocks(self) -> bool:
        return self.enable and self._dump_blocks
    
    @property
    def dump_attn(self) -> bool:
        return self.enable and self._dump_attn

    @property
    def dump_linear(self) -> bool:
        return self.enable and self._dump_linear
    
    @property
    def offline_input_path(self) -> Optional[str]:
        if self.dump_input:
            path = os.path.join(self._base_path, f"step{self._current_step}", "offline_input")
            os.makedirs(path, exist_ok=True)
            return path
        return None
    
    @property
    def init_data_path(self) -> Optional[str]:
        if self.dump_init:
            path = os.path.join(self._base_path, f"step{self._current_step}", "init_data")
            os.makedirs(path, exist_ok=True)
            return path
        return None
    
    @property
    def per_block_result_path(self) -> Optional[str]:
        if self.dump_blocks:
            path = os.path.join(self._base_path, f"step{self._current_step}", "pre_block_result")
            os.makedirs(path, exist_ok=True)
            return path
        return None
    
    @property
    def attn_path(self) -> Optional[str]:
        if self.dump_attn:
            path = os.path.join(self._base_path, f"step{self._current_step}", "sparse_attn")
            os.makedirs(path, exist_ok=True)
            return path
        return None
    
    @property
    def linear_path(self) -> Optional[str]:
        if self.dump_linear:
            path = os.path.join(self._base_path, f"step{self._current_step}", "linear")
            os.makedirs(path, exist_ok=True)
            return path
        return None
    
    @property
    def input_shape(self) -> Optional[str]:
        return self._input_shape


DUMP_CFG = DumpConfig.get_instance()
DUMP_CFG._update_global_paths()

INPUT_SHAPE = DUMP_CFG.input_shape
