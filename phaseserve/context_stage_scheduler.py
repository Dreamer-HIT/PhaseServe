# 上下文阶段调度器 - 负责在上下文处理阶段调度请求的执行
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
import copy  # 导入深拷贝功能
from typing import List, Callable, Tuple  # 导入类型提示

from phaseserve.config import ContextStageSchedConfig, ParallelConfig  # 导入配置类
from phaseserve.logger import init_logger  # 导入日志初始化器
from phaseserve.request import Request, BatchedRequests, MigratingRequest  # 导入请求相关类
from phaseserve.block_manager import BlockManager  # 导入块管理器

# 初始化当前模块的日志记录器
logger = init_logger(__name__)


class ContextStageScheduler(ABC):
    """
    ContextStageScheduler: 上下文调度器的抽象基类。
    
    它应该维护当前系统中的所有请求，并支持两个基本操作：
        - add_request: 将新到达的请求添加到等待队列中
        - get_next_batch_and_pop: 获取上下文阶段的下一批处理请求，并从等待队列中
          移除这些请求。
    
    这个调度器比DecodingStageScheduler更简单，因为一个请求只会被一个上下文阶段处理。      
    """

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """
        向调度器添加一个请求。
        """
        raise NotImplementedError()

    @abstractmethod
    def abort_request(self, request_id: int) -> None:
        """
        从调度器中取消一个请求。
        """
        raise NotImplementedError()

    @abstractmethod
    def get_next_batch_and_pop(self) -> BatchedRequests:
        """
        获取下一次迭代执行的请求批次，并从等待队列中移除这些请求。
        """
        raise NotImplementedError()

    @abstractmethod
    def get_num_waiting_requests(self) -> int:
        """
        获取正在等待处理的请求数量。
        """
        raise NotImplementedError()

    @abstractmethod
    def print_status(self) -> None:
        """
        打印调度器的状态信息。
        """
        raise NotImplementedError()
    
    def on_finish_requests(self, batch: BatchedRequests) -> None:
        """
        当一批请求完成上下文阶段时的回调函数。
        """
        pass
    
    def on_request_migrated(self, migrated_request: MigratingRequest) -> None:
        """
        当一个请求迁移到解码阶段时的回调函数。
        """
        pass
    
    def post_process(self) -> None:
        """
        每次迭代后的后处理。ContextEventLoop会在每次迭代后调用此函数。
        """
        pass


class ContextStageFCFSScheduler(ContextStageScheduler):
    """
    先来先服务（FCFS）上下文阶段调度器。
    
    该调度器实现了基本的先来先服务策略，按照请求到达的顺序进行处理。
    同时考虑了批处理大小、令牌数量和GPU内存块等资源限制。
    """

    def __init__(
        self,
        sched_config: ContextStageSchedConfig, 
        parallel_config: ParallelConfig,
        block_manager: BlockManager):
        """
        初始化FCFS调度器。
        
        参数:
            sched_config: 上下文调度器配置对象
            parallel_config: 并行配置对象
            block_manager: 内存块管理器
        """
        assert (
            sched_config.policy == "fcfs"
        ), f"无法使用策略 {sched_config.policy} 初始化FCFS调度器"
        self.sched_config = sched_config
        # 如果当前批次已满，请求将被放入等待队列
        self.waiting_queue = []
        self.parallel_config: List[Request] = copy.deepcopy(parallel_config)
        self.block_manager = block_manager
        # 已完成上下文阶段但尚未被解码阶段接受的请求
        self.unaccepted_queue: List[Request] = []
        # 正在处理中的请求块数量
        # 在调用get_next_batch_and_pop()时增加
        # 在调用on_finish_requests()时减少
        self.num_on_fly_request_block = 0

    def add_request(self, request: Request) -> None:
        """
        向调度器添加一个请求。
        
        参数:
            request: 要添加的请求对象
        """
        self.waiting_queue.append(request)

    def abort_request(self, request_id: int) -> None:
        """
        从调度器中取消一个请求。
        
        参数:
            request_id: 要取消的请求ID
            
        该方法会在等待队列中查找并移除指定ID的请求。
        """
        for i, request in enumerate(self.waiting_queue):
            if request.request_id == request_id:
                del self.waiting_queue[i]
                return

    def _get_block_needed(self, length: int) -> int:
        """
        计算给定长度的输入需要多少内存块。
        
        参数:
            length: 输入的长度（通常是token数量）
            
        返回:
            所需的内存块数量
        """
        block_size = self.block_manager.cache_config.block_size
        return (length + block_size - 1) // block_size
            
    def get_next_batch_and_pop(self) -> BatchedRequests:
        """
        以FCFS（先来先服务）方式获取上下文阶段的下一批请求，并从等待队列中移除它们。
        
        返回:
            BatchedRequests: 包含选定请求的批处理对象
        """
        next_batch = BatchedRequests()

        def _check_add_to_cur_batch(request: Request) -> bool:
            """
            检查请求是否可以添加到当前批次中。
            
            参数:
                request: 要检查的请求对象
                
            返回:
                布尔值，表示是否可以添加请求
                
            此函数考虑三个限制条件:
            1. 批次大小限制：不超过最大批次大小
            2. 每批次令牌数限制：不超过每批次最大令牌数
            3. GPU内存块限制：不超过可用的GPU内存块数量
            """
            return (
                # 限制1：批次大小
                len(next_batch) < self.sched_config.max_batch_size
            ) and (
                # 限制2：每批次令牌数
                next_batch.get_num_input_tokens()
                + request.get_num_input_tokens()
                <= self.sched_config.max_tokens_per_batch
            ) and (
                # 限制3：GPU内存块
                sum([
                    self._get_block_needed(len(req.prompt_token_ids))
                    for req in next_batch.requests + [request]
                ]) +
                sum([
                    self._get_block_needed(len(req.prompt_token_ids))
                    for req in self.unaccepted_queue
                ]) +
                self.num_on_fly_request_block 
                <= self.block_manager.max_num_gpu_blocks
            )
    
        # 按FCFS顺序处理等待队列中的请求
        while len(self.waiting_queue) > 0:
            request = self.waiting_queue[0]
            if _check_add_to_cur_batch(request):
                next_batch.add_request(request)
                self.waiting_queue.pop(0)
            else:
                # 如果当前请求无法添加到批次中（由于任何限制），停止处理
                break
        
        # 更新正在处理中的请求块数量
        self.num_on_fly_request_block += sum([
            self._get_block_needed(req.get_input_len())
            for req in next_batch.requests
        ])

        return next_batch

    def on_finish_requests(self, batch: BatchedRequests) -> None:
        """
        当一批请求完成上下文阶段处理时的回调函数。
        
        参数:
            batch: 完成处理的批处理请求对象
            
        对于未完成的请求（通常是需要进入解码阶段的请求），
        将其添加到未接受队列中等待迁移到解码阶段。
        同时更新正在处理中的请求块数量。
        """
        for request in batch.requests:
            if not request.is_finished:
                self.unaccepted_queue.append(request)
        
        # 减少正在处理中的请求块数量
        self.num_on_fly_request_block -= sum([
            self._get_block_needed(req.get_input_len())
            for req in batch.requests
        ])

#     def on_finish_requests(self, batch: BatchedRequests) -> None:
#         """
#         当一批请求完成上下文阶段处理时的回调函数。

#         对于未完成的请求（通常是需要进入解码阶段的请求），
#         将其添加到未接受队列中等待迁移到解码阶段。
#         同时更新正在处理中的请求块数量。
#         单阶段模式下，已完成的请求不会再进入unaccepted_queue，并且每轮都清理掉unaccepted_queue中所有已完成的请求。
#         """
#         # 清理unaccepted_queue中所有已完成的请求
#         self.unaccepted_queue = [r for r in self.unaccepted_queue if not r.is_finished]
#         for request in batch.requests:
#             if not request.is_finished:
#                 self.unaccepted_queue.append(request)
#         # 再次清理，确保无多余残留
#         self.unaccepted_queue = [r for r in self.unaccepted_queue if not r.is_finished]
        
#         # 调试：打印unaccepted_queue所有request_id及is_finished状态
#         print("[调试] on_finish_requests后 unaccepted_queue: ", [(r.request_id, r.is_finished) for r in self.unaccepted_queue])
        
#         # 减少正在处理中的请求块数量
#         self.num_on_fly_request_block -= sum([
#             self._get_block_needed(req.get_input_len())
#             for req in batch.requests
#         ])
    
    def on_request_migrated(self, migrated_request: MigratingRequest) -> None:
        """
        当一个请求迁移到解码阶段时的回调函数。
        
        参数:
            migrated_request: 已迁移的请求对象
            
        从未接受队列中移除已成功迁移到解码阶段的请求。
        """
        for i, request in enumerate(self.unaccepted_queue):
            if request.request_id == migrated_request.req.request_id:
                del self.unaccepted_queue[i]
                return
            
    def get_num_waiting_requests(self) -> int:
        """
        获取正在等待处理的请求数量。
        
        返回:
            等待队列中的请求数量
        """
        return len(self.waiting_queue)

    def __repr__(self) -> str:
        """
        返回调度器的字符串表示。
        
        返回:
            包含关键配置参数的字符串
        """
        return (
            f"FCFS(max_batch_size={self.sched_config.max_batch_size}, "
            f"max_tokens_per_batch={self.sched_config.max_tokens_per_batch})"
        )
    
    def print_status(self) -> None:
        """
        打印调度器的当前状态信息。
        
        输出包括：等待请求数量、已完成但未被接受的请求数量，以及
        正在处理中的请求占用的内存块数量。
        """
        logger.info(f"(context) {len(self.waiting_queue)}个等待中, {len(self.unaccepted_queue)}个已完成但未被接受, {self.num_on_fly_request_block}个内存块被处理中的请求占用")


class ContextStageSJFScheduler(ContextStageScheduler):
    """
    最短作业优先（SJF）上下文阶段调度器。
    
    该调度器实现了最短作业优先策略，每次选择请求长度最小的请求进行处理。
    这种策略可以减少平均等待时间，提高系统吞吐量。
    同时也考虑了批处理大小、令牌数量和GPU内存块等资源限制。
    """

    def __init__(
        self,
        sched_config: ContextStageSchedConfig, 
        parallel_config: ParallelConfig,
        block_manager: BlockManager):
        """
        初始化SJF调度器。
        
        参数:
            sched_config: 上下文调度器配置对象
            parallel_config: 并行配置对象
            block_manager: 内存块管理器
        """
        assert (
            sched_config.policy == "sjf"
        ), f"无法使用策略 {sched_config.policy} 初始化SJF调度器"
        self.sched_config = sched_config
        # 如果当前批次已满，请求将被放入等待队列
        self.waiting_queue = []  # 这里不需要排序，每次选择时动态找最短的
        self.parallel_config: List[Request] = copy.deepcopy(parallel_config)
        self.block_manager = block_manager
        # 已完成上下文阶段但尚未被解码阶段接受的请求
        self.unaccepted_queue: List[Request] = []
        # 正在处理中的请求块数量
        # 在调用get_next_batch_and_pop()时增加
        # 在调用on_finish_requests()时减少
        self.num_on_fly_request_block = 0

    def add_request(self, request: Request) -> None:
        """
        向调度器添加一个请求。
        
        参数:
            request: 要添加的请求对象
        """
        self.waiting_queue.append(request)

    def abort_request(self, request_id: int) -> None:
        """
        从调度器中取消一个请求。
        
        参数:
            request_id: 要取消的请求ID
            
        该方法会在等待队列中查找并移除指定ID的请求。
        """
        for i, request in enumerate(self.waiting_queue):
            if request.request_id == request_id:
                del self.waiting_queue[i]
                return

    def _get_block_needed(self, length: int) -> int:
        """
        计算给定长度的输入需要多少内存块。
        
        参数:
            length: 输入的长度（通常是token数量）
            
        返回:
            所需的内存块数量
        """
        block_size = self.block_manager.cache_config.block_size
        return (length + block_size - 1) // block_size
            
    def get_next_batch_and_pop(self) -> BatchedRequests:
        """
        以SJF（最短作业优先）方式获取上下文阶段的下一批请求，并从等待队列中移除它们。
        
        返回:
            BatchedRequests: 包含选定请求的批处理对象
        """
        next_batch = BatchedRequests()

        def _check_add_to_cur_batch(request: Request) -> bool:
            """
            检查请求是否可以添加到当前批次中。
            
            参数:
                request: 要检查的请求对象
                
            返回:
                布尔值，表示是否可以添加请求
                
            此函数考虑三个限制条件:
            1. 批次大小限制：不超过最大批次大小
            2. 每批次令牌数限制：不超过每批次最大令牌数
            3. GPU内存块限制：不超过可用的GPU内存块数量
            """
            return (
                # 限制1：批次大小
                len(next_batch) < self.sched_config.max_batch_size
            ) and (
                # 限制2：每批次令牌数
                next_batch.get_num_input_tokens()
                + request.get_num_input_tokens()
                <= self.sched_config.max_tokens_per_batch
            ) and (
                # 限制3：GPU内存块
                sum([
                    self._get_block_needed(len(req.prompt_token_ids))
                    for req in next_batch.requests + [request]
                ]) +
                sum([
                    self._get_block_needed(len(req.prompt_token_ids))
                    for req in self.unaccepted_queue
                ]) +
                self.num_on_fly_request_block 
                <= self.block_manager.max_num_gpu_blocks
            )
    
        # 按SJF顺序处理等待队列中的请求 - 每次选择请求长度最小的请求
        while len(self.waiting_queue) > 0:
            # 找到等待队列中长度最小的请求
            shortest_idx = 0
            for i in range(1, len(self.waiting_queue)):
                if len(self.waiting_queue[i].prompt_token_ids) < len(self.waiting_queue[shortest_idx].prompt_token_ids):
                    shortest_idx = i
            
            request = self.waiting_queue[shortest_idx]
            if _check_add_to_cur_batch(request):
                next_batch.add_request(request)
                del self.waiting_queue[shortest_idx]  # 从等待队列中移除
            else:
                # 如果当前请求无法添加到批次中（由于任何限制），停止处理
                break
        
        # 更新正在处理中的请求块数量
        self.num_on_fly_request_block += sum([
            self._get_block_needed(req.get_input_len())
            for req in next_batch.requests
        ])

        return next_batch

    def on_finish_requests(self, batch: BatchedRequests) -> None:
        """
        当一批请求完成上下文阶段处理时的回调函数。
        
        参数:
            batch: 完成处理的批处理请求对象
            
        对于未完成的请求（通常是需要进入解码阶段的请求），
        将其添加到未接受队列中等待迁移到解码阶段。
        同时更新正在处理中的请求块数量。
        """
        for request in batch.requests:
            if not request.is_finished:
                self.unaccepted_queue.append(request)
        
        # 减少正在处理中的请求块数量
        self.num_on_fly_request_block -= sum([
            self._get_block_needed(req.get_input_len())
            for req in batch.requests
        ])
    
    def on_request_migrated(self, migrated_request: MigratingRequest) -> None:
        """
        当一个请求迁移到解码阶段时的回调函数。
        
        参数:
            migrated_request: 已迁移的请求对象
            
        从未接受队列中移除已成功迁移到解码阶段的请求。
        """
        for i, request in enumerate(self.unaccepted_queue):
            if request.request_id == migrated_request.req.request_id:
                del self.unaccepted_queue[i]
                return
            
    def get_num_waiting_requests(self) -> int:
        """
        获取正在等待处理的请求数量。
        
        返回:
            等待队列中的请求数量
        """
        return len(self.waiting_queue)

    def __repr__(self) -> str:
        """
        返回调度器的字符串表示。
        
        返回:
            包含关键配置参数的字符串
        """
        return (
            f"SJF(max_batch_size={self.sched_config.max_batch_size}, "
            f"max_tokens_per_batch={self.sched_config.max_tokens_per_batch})"
        )
    
    def print_status(self) -> None:
        """
        打印调度器的当前状态信息。
        
        输出包括：等待请求数量、已完成但未被接受的请求数量，以及
        正在处理中的请求占用的内存块数量。
        """
        logger.info(f"(context) {len(self.waiting_queue)}个等待中, {len(self.unaccepted_queue)}个已完成但未被接受, {self.num_on_fly_request_block}个内存块被处理中的请求占用")


def get_context_stage_scheduler(
    sched_config: ContextStageSchedConfig,
    parallel_config: ParallelConfig,
    block_manager: BlockManager
) -> ContextStageScheduler:
    """
    上下文阶段调度器的工厂函数。
    
    根据配置中指定的策略创建并返回相应的上下文阶段调度器实例。
    
    参数:
        sched_config: 调度器配置对象
        parallel_config: 并行配置对象
        block_manager: 内存块管理器
        
    返回:
        创建的上下文阶段调度器实例
        
    异常:
        NotImplementedError: 当指定的调度策略不受支持时抛出
    """
    if sched_config.policy == "fcfs":
        return ContextStageFCFSScheduler(sched_config, parallel_config, block_manager)
    elif sched_config.policy == "sjf":
        return ContextStageSJFScheduler(sched_config, parallel_config, block_manager)
    else:
        raise NotImplementedError(f"未知的上下文调度器策略: {sched_config.policy}")
