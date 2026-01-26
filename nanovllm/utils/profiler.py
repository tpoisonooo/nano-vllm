import torch
import torch.nn as nn
from collections import defaultdict
import time
import threading

class ThreadSafeLayerProfiler:
    def __init__(self):
        self._stats = defaultdict(lambda: {"count": 0, "total_time": 0.0, "name": ""})
        self._hooks = []
        self._enabled = True
        
        # 关键点：每个线程有自己的 start_times 字典
        self._thread_local = threading.local()
    
    @property
    def _start_times(self):
        """线程安全的属性访问"""
        if not hasattr(self._thread_local, 'start_times'):
            self._thread_local.start_times = {}
        return self._thread_local.start_times
    
    def _pre_hook(self, module, input):
        if not self._enabled:
            return
        # 现在每个线程独立存自己的时间戳
        self._start_times[id(module)] = time.perf_counter()
    
    def _post_hook(self, module, input, output):
        if not self._enabled:
            return
        module_id = id(module)
        if module_id in self._start_times:
            elapsed = time.perf_counter() - self._start_times.pop(module_id)
            name = getattr(module, '_layer_name', f"{type(module).__name__}_{module_id}")
            self._stats[module_id]["name"] = name
            self._stats[module_id]["count"] += 1
            self._stats[module_id]["total_time"] += elapsed
    
    def register_model(self, model: nn.Module, prefix=""):
        """递归给所有子模块注册 hook"""
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            module._layer_name = full_name  # 把名字存到模块上
            
            # 注册 pre 和 post hook
            pre_hook = module.register_forward_pre_hook(self._pre_hook)
            post_hook = module.register_forward_hook(self._post_hook)
            self._hooks.extend([pre_hook, post_hook])
            
            # 递归处理子模块
            self.register_model(module, full_name)
    
    def get_stats(self, sort_by_time=True):
        """返回统计信息，默认按耗时排序"""
        stats_list = [
            {
                "name": v["name"],
                "count": v["count"],
                "total_time": v["total_time"],
                "avg_time": v["total_time"] / max(v["count"], 1)
            }
            for v in self._stats.values()
        ]
        if sort_by_time:
            stats_list.sort(key=lambda x: x["total_time"], reverse=True)
        return stats_list
    
    def print_stats(self, top_k=-1, min_level=3):
        """打印统计结果"""
        print(f"\n{'='*80}")
        print(f"{'Layer Name':<40} {'Count':<8} {'Total (ms)':<12} {'Avg (ms)':<12}")
        print(f"{'='*80}")
        
        for stat in self.get_stats()[:top_k]:
            stat_name = stat['name']
            level_count = stat_name.count('.')
            if level_count < min_level:
                continue
            print(
                f"{stat['name']:<40} {stat['count']:<8} "
                f"{stat['total_time']*1000:<12.3f} {stat['avg_time']*1000:<12.3f}"
            )
        print(f"{'='*80}\n")
    
    def reset(self):
        """清空统计"""
        self._stats.clear()
        self._start_times.clear()
    
    def remove_hooks(self):
        """移除所有 hook，清理资源"""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()