from abc import ABC, abstractmethod
import copy
from typing import List, Callable, Tuple
import warnings
import torch

from phaseserve.config import ParallelConfig, DecodingStageSchedConfig
from phaseserve.logger import init_logger
from phaseserve.request import Request, BatchedRequests, MigratingRequest
from phaseserve.profiling import ProfilingDatabase
from phaseserve.block_manager import BlockManager, BlockLocation

# 初始化日志记录器
logger = init_logger(__name__)


class DecodingStageScheduler(ABC):
    """解码阶段调度器的抽象基类。
    该调度器负责维护系统中的所有请求及其运行时统计信息，这些信息用于调度决策。
    在每次迭代开始前，LLMEngine会调用get_next_batch()方法获取下一次迭代的
    BatchedRequests对象。在每次迭代结束后，LLMEngine会调用pop_finished_requests()
    方法获取当前迭代中已完成的请求。
    """
    
    @abstractmethod
    def add_request(self, request: MigratingRequest) -> None:
        """
        向调度器添加一个请求。
        注意：调度器可能会选择主动迁移请求以提高性能。
        
        参数:
            request: 要添加的迁移请求对象
        """
        raise NotImplementedError()

    @abstractmethod
    def abort_request(self, request_id: int) -> None:
        """
        从调度器中终止一个请求。
        
        参数:
            request_id: 要终止的请求ID
        """
        raise NotImplementedError()

    @abstractmethod
    def get_next_batch(self) -> BatchedRequests:
        """
        获取下一次迭代执行的请求批次。
        
        返回:
            BatchedRequests: 包含下一批处理请求的对象
        """
        raise NotImplementedError()

    @abstractmethod
    def pop_finished_requests(self) -> List[Request]:
        """
        从调度器中弹出已完成的请求。
        
        返回:
            List[Request]: 已完成请求的列表
        """
        raise NotImplementedError()

    @abstractmethod
    def get_total_num_requests(self) -> int:
        """
        获取系统中请求的总数量。
        
        返回:
            int: 系统中的总请求数
        """
        raise NotImplementedError()

    @abstractmethod
    def get_processing_num_requests(self) -> int:
        """
        获取正在处理中的请求数量。
        
        返回:
            int: 正在处理中的请求数
        """
        raise NotImplementedError()

    @abstractmethod
    def get_waiting_num_requests(self) -> int:
        """
        获取正在等待处理的请求数量。
        
        返回:
            int: 等待处理的请求数
        """
        raise NotImplementedError()

    @abstractmethod
    def print_status(self) -> None:
        """
        打印调度器的状态信息。
        """
        raise NotImplementedError()
    
    async def post_process(self) -> None:
        """
        每次迭代后的后处理操作。
        
        此为异步方法，允许在每次迭代后执行必要的清理或准备工作。
        基类中提供空实现，子类可以根据需要重写。
        """
        pass


class DecodingStageFCFSScheduler(DecodingStageScheduler):
    """先来先服务（FCFS）解码阶段调度器。
    注意：它支持流水线并行。它维护着#pp个不相交的批次，这些批次在流水线中正在执行。
    注意：请求要么在等待队列(waiting_queue)中，要么在批处理队列(batch_queues)中，
    一个请求在同一时刻只能在一个队列中。
    """

    def __init__(
        self,
        sched_config: DecodingStageSchedConfig,
        parallel_config: ParallelConfig,
        block_manager: BlockManager,
        engine_migrate_block_callback: Callable,
    ):
        """
        初始化FCFS解码阶段调度器。
        
        参数:
            sched_config: 解码阶段调度配置
            parallel_config: 并行配置
            block_manager: 内存块管理器
            engine_migrate_block_callback: 引擎迁移内存块的回调函数
        """
        assert (
            sched_config.policy == "fcfs"
        ), f"无法使用策略 {sched_config.policy} 初始化FCFS调度器"
        self.sched_config = sched_config
        # 如果请求尚未被接受（即它仍然驻留在"桥接"队列中，
        # 且其内存块仍在上下文阶段引擎一侧），则会被放入未接受队列。
        self.unaccepted_queue: List[MigratingRequest] = []
        # 如果当前批次已满，请求将被放入等待队列。
        self.waiting_queue: List[Request] = []
        # 如果一个请求之前在batch_queues中，但被换出，它将被放入交换队列。
        self.swapped_queue: List[Request] = []
        # 由于使用了流水线并行，系统中有多个批次。
        self.cur_index = -1
        self.batch_queues = [
            BatchedRequests() for i in range(parallel_config.pipeline_parallel_size)
        ]
        self.parallel_config = copy.deepcopy(parallel_config)
        self.block_manager = block_manager
        self.engine_migrate_block_callback = engine_migrate_block_callback

    def _get_block_needed(self, length: int) -> int:
        """
        计算给定长度的输入需要的内存块数量。
        
        参数:
            length: 输入长度（通常是token数量）
            
        返回:
            int: 所需的内存块数量
        """
        block_size = self.block_manager.cache_config.block_size
        return (length + block_size - 1) // block_size
        
    def _check_add_to_cur_batch(self, request: Request) -> bool:
        """
        检查请求是否可以添加到当前批次中。
        
        参数:
            request: 要检查的请求对象
            
        返回:
            bool: 如果可以添加则为True，否则为False
            
        检查三个条件:
        1. 批次大小限制
        2. 每批次令牌数限制
        3. GPU内存块限制
        """
        return (
            # 限制1：批次大小
            len(self.batch_queues[self.cur_index]) < self.sched_config.max_batch_size
        ) and (
            # 限制2：每批次令牌数
            self.batch_queues[self.cur_index].get_num_input_tokens()
            + request.get_num_input_tokens()
            <= self.sched_config.max_tokens_per_batch
        ) and (
            # 限制3：GPU内存块
            sum([
                sum([
                    self._get_block_needed(len(req.prompt_token_ids) + req.get_output_len())
                    for req in self.batch_queues[index].requests
                ])
                for index in range(self.parallel_config.pipeline_parallel_size)
            ]) + sum([
                self._get_block_needed(len(req.prompt_token_ids))
                for req in self.waiting_queue
            ]) + self._get_block_needed(request.get_input_len() + request.get_output_len()) \
                <= self.block_manager.max_num_gpu_blocks
        )

    # 请求相关方法
    async def add_request(self, migrating_req: MigratingRequest) -> None:
        """
        向调度器添加一个请求。
        
        参数:
            migrating_req: 要添加的迁移请求对象
            
        这里采用简单的方法：接受任何传入的请求。
        请求首先会被添加到未接受队列中，然后在post_process中考虑接受。
        """
        self.unaccepted_queue.append(migrating_req)

    def abort_request(self, request_id: int) -> None:
        """
        从调度器中终止一个请求。
        
        参数:
            request_id: 要终止的请求ID
            
        流程:
        1. 首先扫描所有批次队列
        2. 如找到匹配ID的请求，将其标记为已完成
        3. 如果批次队列中没有，则检查等待队列
        """
        # 扫描当前批次
        for queue in self.batch_queues:
            for _, request in enumerate(queue.requests):
                if request.request_id == request_id:
                    # 这个请求可能正在被模型处理，
                    # 所以直接从当前批次中删除是不安全的。
                    # 将其标记为已完成将最终释放它占用的资源。
                    request.is_finished = True
                    return

        # 扫描等待队列
        for i, request in enumerate(self.waiting_queue):
            if request.request_id == request_id:
                del self.waiting_queue[i]
                return

    def _get_last_stage_batch(self) -> BatchedRequests:
        """
        获取流水线中最后一个阶段的批次。
        
        由于使用流水线并行，需要确定最后一个阶段的索引。
        在FCFS模型中，最后一个阶段是当前索引的下一个（循环）。
        
        返回:
            BatchedRequests: 最后阶段的批处理请求对象
        """
        last_stage_index = (
            self.cur_index + 1
        ) % self.parallel_config.pipeline_parallel_size
        return self.batch_queues[last_stage_index]

    def pop_finished_requests(self) -> List[Request]:
        """
        从调度器中弹出已完成的请求。
        
        这个方法会从流水线的最后一个阶段获取并返回已完成的请求。
        
        返回:
            List[Request]: 已完成请求的列表
        """
        return self._get_last_stage_batch().pop_finished_requests()

    def get_next_batch(self) -> BatchedRequests:
        """
        获取下一次迭代执行的请求批次。
        
        该方法实现了FCFS解码阶段的关键逻辑：
        1. 更新当前处理的流水线阶段索引
        2. 检查GPU内存块是否足够，必要时进行换出
        3. 尝试添加新请求到当前批次
        
        返回:
            BatchedRequests: 下一批要处理的请求
        """
        # 更新流水线的当前索引
        self.cur_index = (
            self.cur_index + 1
        ) % self.parallel_config.pipeline_parallel_size

        # 检查GPU上的内存块是否足够处理下一批请求
        # 如果不够，则换出最后一个请求
        while sum([
            sum([
                self._get_block_needed(req.get_input_len() + req.get_output_len())
                for req in self.batch_queues[index].requests
            ])
            for index in range(self.parallel_config.pipeline_parallel_size)
        ]) + sum([
            self._get_block_needed(req.get_input_len())
            for req in self.waiting_queue
        ]) > self.block_manager.max_num_gpu_blocks:
            logger.info("GPU内存块不足。触发换出操作")
            request = self.batch_queues[self.cur_index].requests.pop(-1)
            self.swapped_queue.append(request)
            self.block_manager.swap_out_requests([request])

        # 尝试添加新请求。优先考虑交换队列中的请求。
        while len(self.swapped_queue) > 0 or len(self.waiting_queue) > 0:
            if len(self.swapped_queue) > 0:
                # 优先处理之前被换出的请求
                request = self.swapped_queue[0]
                if self._check_add_to_cur_batch(request):
                    logger.info("触发换入操作")
                    self.block_manager.swap_in_requests([request])
                    self.batch_queues[self.cur_index].add_request(request)
                    self.swapped_queue.pop(0)
                else:
                    # 如果当前请求无法添加（由于资源限制），则停止处理
                    break
            else:
                # 处理等待队列中的请求
                request = self.waiting_queue[0]
                if self._check_add_to_cur_batch(request):
                    self.batch_queues[self.cur_index].add_request(request)
                    self.waiting_queue.pop(0)
                else:
                    # 如果当前请求无法添加（由于资源限制），则停止处理
                    break
        return self.batch_queues[self.cur_index]

    # 获取函数
    def get_total_num_requests(self) -> int:
        """
        获取系统中请求的总数量。
        
        返回:
            int: 系统中正在处理和等待处理的请求总数
        """
        return self.get_processing_num_requests() + self.get_waiting_num_requests()

    def get_processing_num_requests(self) -> int:
        """
        获取正在处理中的请求数量。
        
        计算所有批处理队列中的请求总数。
        
        返回:
            int: 正在处理中的请求数
        """
        num = 0
        for batch in self.batch_queues:
            num = num + len(batch.requests)
        return num

    def get_waiting_num_requests(self) -> int:
        """
        获取正在等待处理的请求数量。
        
        返回:
            int: 等待队列中的请求数
        """
        return len(self.waiting_queue)

    def __repr__(self) -> str:
        """
        返回调度器的字符串表示。
        
        返回:
            str: 包含关键配置参数的字符串
        """
        return (
            f"FCFS(max_batch_size={self.sched_config.max_batch_size}, "
            f"max_tokens_per_batch={self.sched_config.max_tokens_per_batch})"
        )
    
    def print_status(self) -> None:
        """
        打印调度器的状态信息。
        
        输出未接受、等待处理和正在处理的请求数量。
        """
        logger.info(f"(解码) {len(self.unaccepted_queue)}个未接受, {len(self.waiting_queue)}个等待中, {self.get_processing_num_requests()}个处理中")

    async def post_process(self) -> None:
        """
        每次迭代后的后处理操作。
        
        此方法处理未接受队列中的请求，决定哪些请求可以被接受并迁移到解码阶段。
        判断标准基于当前等待队列的内存块占用情况和可用GPU内存块数量。
        """
        def should_accept(migrating_req: MigratingRequest) -> bool:
            """
            判断是否应该接受一个迁移请求。
            
            参数:
                migrating_req: 要判断的迁移请求
                
            返回:
                bool: 如果应该接受返回True，否则返回False
                
            判断标准:
            1. 等待队列的内存块占用不超过阈值
            2. 请求所需的内存块数量不超过可用GPU内存块
            """
            # return sum([self._get_block_needed(len(req.prompt_token_ids))
            #             for req in self.waiting_queue
            #         ]) < self.block_manager.max_num_gpu_blocks * self.sched_config.waiting_block_prop_threshold \
            #         and self._get_block_needed(len(migrating_req.req.prompt_token_ids)) <= self.block_manager.get_num_avail_gpu_blocks()
            return sum([self._get_block_needed(len(req.prompt_token_ids))
                        for req in self.waiting_queue
                    ]) < self.block_manager.max_num_gpu_blocks * 0.4 \
                    and self._get_block_needed(len(migrating_req.req.prompt_token_ids)) <= self.block_manager.get_num_avail_gpu_blocks()
        
        # 尝试接受未接受队列中的请求
        while len(self.unaccepted_queue) > 0:
            migrating_req = self.unaccepted_queue[0]
            if should_accept(migrating_req):
                self.unaccepted_queue.pop(0)
                # 调用回调函数迁移内存块
                await self.engine_migrate_block_callback(migrating_req)
                # 将请求添加到等待队列
                self.waiting_queue.append(migrating_req.req)
            else:
                # 如果当前请求无法接受，停止处理
                break




class DecodingStageMLFQScheduler(DecodingStageScheduler):
    """多级反馈队列(MLFQ)解码阶段调度器。
    
    根据请求的迭代次数将其分配到不同级别的队列中，优先处理更低级别的队列中的请求。
    队列级别基于实际负载分析设计:
    - 第1级：处理迭代次数≤100的请求 (占52%)
    - 第2级：处理迭代次数在101-200之间的请求 (占10%)
    - 第3级：处理迭代次数在201-300之间的请求 (占11%)
    - 第4级：处理迭代次数在301-500之间的请求 (占17%)
    - 第5级：处理迭代次数>500的请求 (占10%)
    
    注意：它支持流水线并行。它维护着#pp个不相交的批次，这些批次在流水线中正在执行。
    """

    def __init__(
        self,
        sched_config: DecodingStageSchedConfig,
        parallel_config: ParallelConfig,
        block_manager: BlockManager,
        engine_migrate_block_callback: Callable,
    ):
        """
        初始化MLFQ解码阶段调度器。
        
        参数:
            sched_config: 解码阶段调度配置
            parallel_config: 并行配置
            block_manager: 内存块管理器
            engine_migrate_block_callback: 引擎迁移内存块的回调函数
        """
        assert (
            sched_config.policy == "mlfq"
        ), f"无法使用策略 {sched_config.policy} 初始化MLFQ调度器"
        self.sched_config = sched_config
        
        # 如果请求尚未被接受，则会被放入未接受队列
        self.unaccepted_queue: List[MigratingRequest] = []
        
        # 多级反馈队列，根据实际负载分析设计
        # 索引0对应第1级，索引1对应第2级，以此类推
        self.level_thresholds = [100, 200, 300, 500, float('inf')]  # 各级队列的迭代次数阈值
        self.waiting_queues = [[] for _ in range(len(self.level_thresholds))]  # 各级等待队列
        
        # 如果一个请求之前在batch_queues中，但被换出，它将被放入交换队列
        self.swapped_queue: List[Request] = []
        
        # 使用流水线并行
        self.cur_index = -1
        self.batch_queues = [
            BatchedRequests() for i in range(parallel_config.pipeline_parallel_size)
        ]
        self.parallel_config = copy.deepcopy(parallel_config)
        self.block_manager = block_manager
        self.engine_migrate_block_callback = engine_migrate_block_callback
        
        # 记录每个请求的迭代次数
        self.request_iterations = {}

    def _get_block_needed(self, length: int) -> int:
        """
        计算给定长度的输入需要的内存块数量。
        
        参数:
            length: 输入长度（通常是token数量）
            
        返回:
            int: 所需的内存块数量
        """
        block_size = self.block_manager.cache_config.block_size
        return (length + block_size - 1) // block_size
        
    def _check_add_to_cur_batch(self, request: Request) -> bool:
        """
        检查请求是否可以添加到当前批次中。
        
        参数:
            request: 要检查的请求对象
            
        返回:
            bool: 如果可以添加则为True，否则为False
            
        检查三个条件:
        1. 批次大小限制
        2. 每批次令牌数限制
        3. GPU内存块限制
        """
        return (
            # 限制1：批次大小
            len(self.batch_queues[self.cur_index]) < self.sched_config.max_batch_size
        ) and (
            # 限制2：每批次令牌数
            self.batch_queues[self.cur_index].get_num_input_tokens()
            + request.get_num_input_tokens()
            <= self.sched_config.max_tokens_per_batch
        ) and (
            # 限制3：GPU内存块
            sum([
                sum([
                    self._get_block_needed(len(req.prompt_token_ids) + req.get_output_len())
                    for req in self.batch_queues[index].requests
                ])
                for index in range(self.parallel_config.pipeline_parallel_size)
            ]) + sum([
                self._get_block_needed(len(req.prompt_token_ids))
                for level_queue in self.waiting_queues
                for req in level_queue
            ]) + self._get_block_needed(request.get_input_len() + request.get_output_len()) \
                <= self.block_manager.max_num_gpu_blocks
        )

    def _get_queue_level(self, request: Request) -> int:
        """
        确定请求应该进入哪一级队列。
        
        参数:
            request: 要检查的请求对象
            
        返回:
            int: 请求应该进入的队列级别索引（0-4）
        """
        iterations = self.request_iterations.get(request.request_id, 0)
        
        for i, threshold in enumerate(self.level_thresholds):
            if iterations <= threshold:
                return i
        
        # 如果超出所有阈值，放入最后一级队列
        return len(self.level_thresholds) - 1

    def _update_request_iteration(self, request: Request) -> None:
        """
        更新请求的迭代次数并在必要时重新分配队列级别。
        
        参数:
            request: 要更新的请求对象
            
        返回:
            int: 更新后的迭代次数
        """
        if request.request_id not in self.request_iterations:
            self.request_iterations[request.request_id] = 0
        
        # 当请求完成一次迭代后，更新迭代计数
        self.request_iterations[request.request_id] += 1
        
        # 打印日志记录请求迭代次数
        logger.debug(f"请求 {request.request_id} 当前迭代次数: {self.request_iterations[request.request_id]}")
        
        # 返回更新后的迭代次数
        return self.request_iterations[request.request_id]

    # 请求相关方法
    async def add_request(self, migrating_req: MigratingRequest) -> None:
        """
        向调度器添加一个请求。
        
        参数:
            migrating_req: 要添加的迁移请求对象
        """
        self.unaccepted_queue.append(migrating_req)

    def abort_request(self, request_id: int) -> None:
        """
        从调度器中终止一个请求。
        
        参数:
            request_id: 要终止的请求ID
        """
        # 扫描当前批次
        for queue in self.batch_queues:
            for _, request in enumerate(queue.requests):
                if request.request_id == request_id:
                    # 将其标记为已完成
                    request.is_finished = True
                    # 清理迭代计数记录
                    if request_id in self.request_iterations:
                        del self.request_iterations[request_id]
                    return

        # 扫描各级等待队列
        for level_queue in self.waiting_queues:
            for i, request in enumerate(level_queue):
                if request.request_id == request_id:
                    level_queue.pop(i)
                    # 清理迭代计数记录
                    if request_id in self.request_iterations:
                        del self.request_iterations[request_id]
                    return

    def _get_last_stage_batch(self) -> BatchedRequests:
        """
        获取流水线中最后一个阶段的批次。
        
        返回:
            BatchedRequests: 最后阶段的批处理请求对象
        """
        last_stage_index = (
            self.cur_index + 1
        ) % self.parallel_config.pipeline_parallel_size
        return self.batch_queues[last_stage_index]

    def pop_finished_requests(self) -> List[Request]:
        """
        从调度器中弹出已完成的请求。
        
        返回:
            List[Request]: 已完成请求的列表
        """
        # 获取已完成的请求
        finished_requests = self._get_last_stage_batch().pop_finished_requests()
        
        # 清理已完成请求的迭代计数
        for request in finished_requests:
            if request.request_id in self.request_iterations:
                logger.debug(f"请求 {request.request_id} 完成，最终迭代次数: {self.request_iterations[request.request_id]}")
                del self.request_iterations[request.request_id]
                
        return finished_requests

    def _handle_unfinished_requests(self, batch: BatchedRequests) -> None:
        """
        处理批次中未完成的请求，将它们重新分配到相应级别的等待队列中。
        
        未完成的请求会被放到对应级别等待队列的头部，确保它们在下一次迭代中被优先选择。
        
        参数:
            batch: 包含未完成请求的批处理对象
        """
        # 获取批次中未完成的请求
        unfinished_requests = []
        for request in batch.requests:
            if not request.is_finished:
                unfinished_requests.append(request)
        
        if len(unfinished_requests) > 0:
            logger.debug(f"找到 {len(unfinished_requests)} 个未完成的请求需要重新分配")
                
        # 清空批次
        batch.requests = []
        
        # 重新分配未完成的请求到相应级别的等待队列中
        # 将未完成请求放到队列头部，而不是尾部，以优先处理
        for request in unfinished_requests:
            # 确定请求应该在的队列级别
            level = self._get_queue_level(request)
            logger.debug(f"重新分配请求 {request.request_id} 到级别 {level+1} 队列头部，迭代次数: {self.request_iterations.get(request.request_id, 0)}")
            # 使用insert(0, ...)而不是append(...)，将请求插入到队列头部
            self.waiting_queues[level].insert(0, request)
            
    def get_next_batch(self) -> BatchedRequests:
        """
        获取下一次迭代执行的请求批次。
        
        该方法实现了MLFQ解码阶段的关键逻辑:
        1. 更新当前处理的流水线阶段索引
        2. 对所有批次中的请求更新迭代次数
        3. 将所有批次中的未完成请求重新分配到相应级别的等待队列
        4. 检查GPU内存块是否足够，必要时进行换出
        5. 从较低级别的队列开始，尝试添加新请求到当前批次
        
        返回:
            BatchedRequests: 下一批要处理的请求
        """
        # 输出每个级别队列的请求数量
        queue_sizes = [len(q) for q in self.waiting_queues]
        logger.debug(f"get_next_batch前等待队列情况: L1:{queue_sizes[0]}, L2:{queue_sizes[1]}, L3:{queue_sizes[2]}, L4:{queue_sizes[3]}, L5:{queue_sizes[4]}")
        
        # 更新流水线的当前索引
        self.cur_index = (
            self.cur_index + 1
        ) % self.parallel_config.pipeline_parallel_size
        
        # 处理所有批次中的请求，而不仅仅是当前批次
        for batch_index, batch in enumerate(self.batch_queues):
            # 如果这是当前需要处理的批次，则需要更新所有请求的迭代计数
            if batch_index == self.cur_index:
                for request in batch.requests:
                    # 记录之前的迭代次数
                    old_iterations = self.request_iterations.get(request.request_id, 0)
                    
                    # 更新迭代次数
                    self._update_request_iteration(request)
                    
                    # 判断请求是否跨越了队列级别阈值
                    old_level = 0
                    for i, threshold in enumerate(self.level_thresholds):
                        if old_iterations <= threshold:
                            old_level = i
                            break
                            
                    new_level = self._get_queue_level(request)
                    
                    # 如果跨越了阈值，记录日志
                    if new_level > old_level:
                        logger.info(f"请求 {request.request_id} 从级别 {old_level+1} 升级到级别 {new_level+1}, 迭代次数: {self.request_iterations[request.request_id]}")
            
            # 将批次中的未完成请求重新分配到相应级别的等待队列
            self._handle_unfinished_requests(batch)

        # 检查GPU上的内存块是否足够处理下一批请求
        # 如果不够，则换出最后一个请求
        while sum([
            sum([
                self._get_block_needed(req.get_input_len() + req.get_output_len())
                for req in self.batch_queues[index].requests
            ])
            for index in range(self.parallel_config.pipeline_parallel_size)
        ]) + sum([
            self._get_block_needed(req.get_input_len())
            for level_queue in self.waiting_queues
            for req in level_queue
        ]) > self.block_manager.max_num_gpu_blocks:
            logger.info("GPU内存块不足。触发换出操作")
            request = self.batch_queues[self.cur_index].requests.pop(-1)
            # 在添加到swapped_queue前，更新请求计数
            iterations = self._update_request_iteration(request)
            # 重新计算请求应该在的队列级别
            level = self._get_queue_level(request)
            logger.info(f"换出请求 {request.request_id}，迭代次数: {iterations}，应在级别 {level+1}")
            self.swapped_queue.append(request)
            self.block_manager.swap_out_requests([request])
            
        # 尝试添加新请求。优先考虑交换队列中的请求。
        if len(self.swapped_queue) > 0:
            # 尝试从交换队列添加尽可能多的请求
            swapped_count = 0
            while len(self.swapped_queue) > 0 and swapped_count < self.sched_config.max_batch_size:
                request = self.swapped_queue[0]
                if self._check_add_to_cur_batch(request):
                    logger.info("触发换入操作")
                    self.block_manager.swap_in_requests([request])
                    self.batch_queues[self.cur_index].add_request(request)
                    self.swapped_queue.pop(0)
                    swapped_count += 1
                else:
                    # 如果无法添加当前请求，停止添加
                    break

        # 尝试从各级队列添加请求，直到批次大小达到限制或无法添加更多请求
        level = 0
        while level < len(self.waiting_queues) and len(self.batch_queues[self.cur_index]) < self.sched_config.max_batch_size:
            level_queue = self.waiting_queues[level]
            
            # 尝试在当前级别队列中找到可以添加的请求
            req_index = 0
            added_from_level = False
            
            # 遍历当前级别队列中的请求
            while req_index < len(level_queue):
                if len(self.batch_queues[self.cur_index]) >= self.sched_config.max_batch_size:
                    break
                    
                request = level_queue[req_index]
                
                # 首先检查请求是否在正确的队列级别
                cur_level = self._get_queue_level(request)
                if cur_level != level:
                    logger.info(f"请求 {request.request_id} 迭代次数变化，从级别 {level+1} 移动到级别 {cur_level+1}, 迭代次数: {self.request_iterations.get(request.request_id, 0)}")
                    # 将请求移到正确的队列级别
                    level_queue.pop(req_index)
                    self.waiting_queues[cur_level].append(request)
                    # 不递增req_index，因为我们已经移除了当前索引的元素
                    continue
                
                # 然后检查是否可以添加到当前批次
                if self._check_add_to_cur_batch(request):
                    self.batch_queues[self.cur_index].add_request(request)
                    level_queue.pop(req_index)
                    logger.debug(f"从级别 {level+1} 队列添加请求 {request.request_id}，当前迭代次数: {self.request_iterations.get(request.request_id, 0)}")
                    added_from_level = True
                    # 不递增req_index，因为我们已经移除了当前索引的元素
                else:
                    # 如果当前请求无法添加，尝试下一个请求
                    req_index += 1
            
            # 如果当前级别没有添加任何请求，尝试下一个级别
            if not added_from_level:
                level += 1
            # 如果已经添加了请求，继续尝试从当前级别添加
            # 这样可以确保优先级更高的请求尽可能多地被处理
            
        # 如果批次中还没有请求，并且交换队列中有请求，再尝试添加
        if len(self.batch_queues[self.cur_index]) == 0 and len(self.swapped_queue) > 0:
            request = self.swapped_queue[0]
            if self._check_add_to_cur_batch(request):
                logger.info("触发换入操作")
                self.block_manager.swap_in_requests([request])
                self.batch_queues[self.cur_index].add_request(request)
                self.swapped_queue.pop(0)
                    
        return self.batch_queues[self.cur_index]

    # 获取函数
    def get_total_num_requests(self) -> int:
        """
        获取系统中请求的总数量。
        
        返回:
            int: 系统中正在处理和等待处理的请求总数
        """
        return self.get_processing_num_requests() + self.get_waiting_num_requests()

    def get_processing_num_requests(self) -> int:
        """
        获取正在处理中的请求数量。
        
        计算所有批处理队列中的请求总数。
        
        返回:
            int: 正在处理中的请求数
        """
        num = 0
        for batch in self.batch_queues:
            num = num + len(batch.requests)
        return num

    def get_waiting_num_requests(self) -> int:
        """
        获取正在等待处理的请求数量。
        
        返回:
            int: 所有等待队列中的请求数总和
        """
        return sum(len(level_queue) for level_queue in self.waiting_queues)

    def __repr__(self) -> str:
        """
        返回调度器的字符串表示。
        
        返回:
            str: 包含关键配置参数的字符串
        """
        return (
            f"MLFQ(max_batch_size={self.sched_config.max_batch_size}, "
            f"max_tokens_per_batch={self.sched_config.max_tokens_per_batch}, "
            f"levels={len(self.waiting_queues)})"
        )
    
    def print_status(self) -> None:
        """
        打印调度器的状态信息。
        
        输出未接受、各级等待处理和正在处理的请求数量。
        """
        level_counts = [len(queue) for queue in self.waiting_queues]
        level_status = ", ".join([f"L{i+1}:{count}" for i, count in enumerate(level_counts)])
        logger.info(f"(解码) {len(self.unaccepted_queue)}个未接受, 等待队列[{level_status}], {self.get_processing_num_requests()}个处理中")

    async def post_process(self) -> None:
        """
        每次迭代后的后处理操作。
        
        此方法处理未接受队列中的请求，决定哪些请求可以被接受并迁移到解码阶段。
        同时检查和重新分配各个等待队列中的请求，确保它们在正确的级别队列中。
        """
        def should_accept(migrating_req: MigratingRequest) -> bool:
            """
            判断是否应该接受一个迁移请求。
            
            参数:
                migrating_req: 要判断的迁移请求
                
            返回:
                bool: 如果应该接受返回True，否则返回False
            """
            # return sum([
            #     self._get_block_needed(len(req.prompt_token_ids))
            #     for level_queue in self.waiting_queues
            #     for req in level_queue
            # ]) < self.block_manager.max_num_gpu_blocks * self.sched_config.waiting_block_prop_threshold \
            #     and self._get_block_needed(len(migrating_req.req.prompt_token_ids)) <= self.block_manager.get_num_avail_gpu_blocks()
            return sum([
                self._get_block_needed(len(req.prompt_token_ids))
                for level_queue in self.waiting_queues
                for req in level_queue
            ]) < self.block_manager.max_num_gpu_blocks * 0.4 \
                and self._get_block_needed(len(migrating_req.req.prompt_token_ids)) <= self.block_manager.get_num_avail_gpu_blocks()
            
        # 输出每个级别队列的请求数量
        queue_sizes = [len(q) for q in self.waiting_queues]
        logger.debug(f"当前等待队列情况: L1:{queue_sizes[0]}, L2:{queue_sizes[1]}, L3:{queue_sizes[2]}, L4:{queue_sizes[3]}, L5:{queue_sizes[4]}")
        
        # 尝试接受未接受队列中的请求，最多接受一定数量的请求
        accept_count = 0
        max_accept_per_iteration = 50  # 每次迭代最多接受的请求数量
        
        while len(self.unaccepted_queue) > 0 and accept_count < max_accept_per_iteration:
            migrating_req = self.unaccepted_queue[0]
            if should_accept(migrating_req):
                self.unaccepted_queue.pop(0)
                # 调用回调函数迁移内存块
                await self.engine_migrate_block_callback(migrating_req)
                
                # 将请求添加到最低级别的等待队列（初始请求都从第1级开始）
                self.waiting_queues[0].append(migrating_req.req)
                # 初始化迭代计数
                self.request_iterations[migrating_req.req.request_id] = 0
                
                logger.debug(f"接受请求 {migrating_req.req.request_id} 到级别1队列")
                accept_count += 1
            else:
                # 如果当前请求无法接受，停止处理
                break
                
        # 检查交换队列中的请求，重新计算它们应该在的队列级别
        if len(self.swapped_queue) > 0:
            swapped_requests_to_requeue = []
            for _ in range(len(self.swapped_queue)):
                request = self.swapped_queue.pop(0)
                swapped_requests_to_requeue.append(request)
                
            # 重新分配这些请求到正确的队列级别
            for request in swapped_requests_to_requeue:
                level = self._get_queue_level(request)
                logger.debug(f"将交换队列中的请求 {request.request_id} 重新分配到级别 {level+1} 队列头部，迭代次数: {self.request_iterations.get(request.request_id, 0)}")
                # 使用insert(0, ...)而不是append(...)，将请求插入到队列头部，优先处理被换出的请求
                self.waiting_queues[level].insert(0, request)
                
        # 检查并重新分配各个等待队列中的请求
        for level, level_queue in enumerate(self.waiting_queues):
            requests_to_requeue = []
            for i in range(len(level_queue) - 1, -1, -1):  # 从后向前遍历，避免删除元素时的索引问题
                request = level_queue[i]
                current_level = self._get_queue_level(request)
                if current_level != level:
                    logger.info(f"请求 {request.request_id} 级别不匹配，从级别 {level+1} 移动到级别 {current_level+1}，迭代次数: {self.request_iterations.get(request.request_id, 0)}")
                    # 从当前队列中移除
                    level_queue.pop(i)
                    # 添加到需要重新分配队列的列表中
                    requests_to_requeue.append((request, current_level))
            
            # 重新分配请求
            for request, new_level in requests_to_requeue:
                self.waiting_queues[new_level].append(request)



def get_decoding_stage_scheduler(
    sched_config: DecodingStageSchedConfig,
    parallel_config: ParallelConfig,
    block_manager: BlockManager,
    engine_migrate_block_callback: Callable,
) -> DecodingStageScheduler:
    """
    解码阶段调度器的工厂函数。
    
    根据配置中指定的策略创建并返回相应的解码阶段调度器实例。
    
    参数:
        sched_config: 调度器配置对象
        parallel_config: 并行配置对象
        block_manager: 内存块管理器
        engine_migrate_block_callback: 引擎迁移内存块的回调函数
        
    返回:
        DecodingStageScheduler: 创建的解码阶段调度器实例
        
    异常:
        NotImplementedError: 当指定的调度策略不受支持时抛出
    """
    if sched_config.policy == "fcfs":
        return DecodingStageFCFSScheduler(sched_config, parallel_config, block_manager, engine_migrate_block_callback)
    elif sched_config.policy == "mlfq":
        return DecodingStageMLFQScheduler(sched_config, parallel_config, block_manager, engine_migrate_block_callback)
    else:
        raise NotImplementedError(
            f"不支持的调度器策略: {sched_config.policy}"
        )
