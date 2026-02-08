from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import (
    Any,
    Literal,
    TypedDict,
    TypeVar,
    Callable,
    Optional,
    Dict,
    List,
    AsyncIterator,
)
from .utils import EmbeddingFunc
from .types import KnowledgeGraph
from .constants import (
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_TOP_K,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_HISTORY_TURNS,
    DEFAULT_OLLAMA_MODEL_NAME,
    DEFAULT_OLLAMA_MODEL_TAG,
    DEFAULT_OLLAMA_MODEL_SIZE,
    DEFAULT_OLLAMA_CREATED_AT,
    DEFAULT_OLLAMA_DIGEST,
)

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

# 当 LightRAG 通过 Ollama 兼容接口对外服务时，模拟 Ollama 模型信息
class OllamaServerInfos:
    """
    LightRAG 通过 Ollama 兼容接口对外服务时，模拟 Ollama 模型信息
    """
    def __init__(self, name=None, tag=None):
        self._lightrag_name = name or os.getenv(
            "OLLAMA_EMULATING_MODEL_NAME", DEFAULT_OLLAMA_MODEL_NAME
        )
        self._lightrag_tag = tag or os.getenv(
            "OLLAMA_EMULATING_MODEL_TAG", DEFAULT_OLLAMA_MODEL_TAG
        )
        self.LIGHTRAG_SIZE = DEFAULT_OLLAMA_MODEL_SIZE
        self.LIGHTRAG_CREATED_AT = DEFAULT_OLLAMA_CREATED_AT
        self.LIGHTRAG_DIGEST = DEFAULT_OLLAMA_DIGEST

    # @property 装饰器
    # 将 LIGHTRAG_NAME 方法转换为只读属性
    # 允许通过 obj.LIGHTRAG_NAME 直接访问，而不需要 obj.LIGHTRAG_NAME()
    # 读取时返回私有属性 _lightrag_name 的值
    @property
    def LIGHTRAG_NAME(self):
        return self._lightrag_name

    # @LIGHTRAG_NAME.setter 装饰器
    # 定义属性的赋值行为
    # 允许通过 obj.LIGHTRAG_NAME = value 修改值
    # 修改时更新私有属性 _lightrag_name
    @LIGHTRAG_NAME.setter
    def LIGHTRAG_NAME(self, value):
        self._lightrag_name = value

    @property
    def LIGHTRAG_TAG(self):
        return self._lightrag_tag

    @LIGHTRAG_TAG.setter
    def LIGHTRAG_TAG(self, value):
        self._lightrag_tag = value

    @property
    def LIGHTRAG_MODEL(self):
        return f"{self._lightrag_name}:{self._lightrag_tag}"


class TextChunkSchema(TypedDict):
    # 块的token数
    tokens: int
    # 块的文本内容
    content: str
    # 所属文档ID
    full_doc_id: str
    # 块在文档中的顺序
    chunk_order_index: int

# 泛型，TypeVar 允许你定义一个变量，该变量可以代表任何类型或一组类型
# 可以避免Any带来的类型检查缺失问题
T = TypeVar("T")

# 用户查询时的核心配置类
@dataclass
class QueryParam:
    """Configuration parameters for query execution in LightRAG."""

    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix"
    """Specifies the retrieval mode:
    - "local": Focuses on context-dependent information.  本地检索 先查实体 再找边
    - "global": Utilizes global knowledge.  全局检索    先查边再找实体
    - "hybrid": Combines local and global retrieval methods.  混合本地+全局检索
    - "naive": Performs a basic search without advanced techniques.  基础向量搜索
    - "mix": Integrates knowledge graph and vector retrieval.  混合知识图谱+向量检索
    - "bypass": Directly uses LLM without retrieval.  直接使用LLM自己的能力进行回答，不进行任何检索
    """

    # 是否只需要返回检索到的上下文，不生成回答
    only_need_context: bool = False
    """If True, only returns the retrieved context without generating a response."""

    # 是否只需要生成的提示词，而不需要生成回答
    only_need_prompt: bool = False
    """If True, only returns the generated prompt without producing a response."""

    # 回答格式
    response_type: str = "Multiple Paragraphs"
    """Defines the response format. Examples: 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'."""

    # 是否流式输出
    stream: bool = False
    """If True, enables streaming output for real-time responses."""

    # 检索的实体/关系数量
    top_k: int = int(os.getenv("TOP_K", str(DEFAULT_TOP_K)))
    """Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode."""

    # 检索的文本块数量
    chunk_top_k: int = int(os.getenv("CHUNK_TOP_K", str(DEFAULT_CHUNK_TOP_K)))
    """Number of text chunks to retrieve initially from vector search and keep after reranking.
    If None, defaults to top_k value.
    """

    # 实体上下文最大 token 数
    max_entity_tokens: int = int(
        os.getenv("MAX_ENTITY_TOKENS", str(DEFAULT_MAX_ENTITY_TOKENS))
    )
    """Maximum number of tokens allocated for entity context in unified token control system."""

    # 关系上下文最大 token 数
    max_relation_tokens: int = int(
        os.getenv("MAX_RELATION_TOKENS", str(DEFAULT_MAX_RELATION_TOKENS))
    )
    """Maximum number of tokens allocated for relationship context in unified token control system."""

    # 总上下文最大 token 数
    max_total_tokens: int = int(
        os.getenv("MAX_TOTAL_TOKENS", str(DEFAULT_MAX_TOTAL_TOKENS))
    )
    """Maximum total tokens budget for the entire query context (entities + relations + chunks + system prompt)."""

    # 高级关键词列表
    hl_keywords: list[str] = field(default_factory=list)
    """List of high-level keywords to prioritize in retrieval."""

    # 低级关键词列表
    ll_keywords: list[str] = field(default_factory=list)
    """List of low-level keywords to refine retrieval focus."""

    # History mesages is only send to LLM for context, not used for retrieval
    # 会话历史消息，仅发送给LLM作为上下文，不用于检索
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    """Stores past conversation history to maintain context.
    Format: [{"role": "user/assistant", "content": "message"}].
    """

    # TODO: deprecated. No longer used in the codebase, all conversation_history messages is send to LLM
    history_turns: int = int(os.getenv("HISTORY_TURNS", str(DEFAULT_HISTORY_TURNS)))
    """Number of complete conversation turns (user-assistant pairs) to consider in the response context."""

    # LLM模型函数覆盖，如果提供则使用此模型函数而不是全局模型函数
    model_func: Callable[..., object] | None = None
    """Optional override for the LLM model function to use for this specific query.
    If provided, this will be used instead of the global model function.
    This allows using different models for different query modes.
    """
    # 用户自定义提示词
    user_prompt: str | None = None
    """User-provided prompt for the query.
    Addition instructions for LLM. If provided, this will be inject into the prompt template.
    It's purpose is the let user customize the way LLM generate the response.
    """
    # 是否启用重新排序
    enable_rerank: bool = os.getenv("RERANK_BY_DEFAULT", "true").lower() == "true"
    """Enable reranking for retrieved text chunks. If True but no rerank model is configured, a warning will be issued.
    Default is True to enable reranking when rerank model is available.
    """
    # 是否在响应中包含参考文献列表
    include_references: bool = False
    """If True, includes reference list in the response for supported endpoints.
    This parameter controls whether the API response includes a references field
    containing citation information for the retrieved content.
    """

# 存储命名空间抽象基类
@dataclass
class StorageNameSpace(ABC):
    # 命名空间（如 "entities", "chunks"）
    namespace: str
    # 工作空间路径（用于多实例隔离）
    workspace: str
    # 全局配置
    global_config: dict[str, Any]

    # 初始化存储
    async def initialize(self):
        """Initialize the storage"""
        pass
    # 最终化存储
    async def finalize(self):
        """Finalize the storage"""
        pass
    
    # 索引完成后回调（提交数据）
    @abstractmethod
    async def index_done_callback(self) -> None:
        """索引完成后回调（提交数据）
        Commit the storage operations after indexing"""

    # 删除所有数据
    @abstractmethod
    async def drop(self) -> dict[str, str]:
        """Drop all data from storage and clean up resources

        This abstract method defines the contract for dropping all data from a storage implementation.
        Each storage type must implement this method to:
        1. Clear all data from memory and/or external storage
        2. Remove any associated storage files if applicable
        3. Reset the storage to its initial state
        4. Handle cleanup of any resources
        5. Notify other processes if necessary
        6. This action should persistent the data to disk immediately.

        Returns:
            dict[str, str]: Operation status and message with the following format:
                {
                    "status": str,  # "success" or "error"
                    "message": str  # "data dropped" on success, error details on failure
                }

        Implementation specific:
        - On success: return {"status": "success", "message": "data dropped"}
        - On failure: return {"status": "error", "message": "<error details>"}
        - If not supported: return {"status": "error", "message": "unsupported"}
        """

# 向量存储抽象基类
@dataclass
class BaseVectorStorage(StorageNameSpace, ABC):
    # 嵌入函数
    embedding_func: EmbeddingFunc
    # 相似度阈值
    cosine_better_than_threshold: float = field(default=0.2)
    # 元数据字段
    meta_fields: set[str] = field(default_factory=set)

    # 验证嵌入函数是否存在
    def _validate_embedding_func(self):
        """Validate that embedding_func is provided.

        This method should be called at the beginning of __post_init__
        in all vector storage implementations.

        Raises:
            ValueError: If embedding_func is None
        """
        if self.embedding_func is None:
            raise ValueError(
                "embedding_func is required for vector storage. "
                "Please provide a valid EmbeddingFunc instance."
            )

    # 从嵌入函数 生成集合/表后缀  就是组合模型名称和维度
    def _generate_collection_suffix(self) -> str | None:
        """Generates collection/table suffix from embedding_func.

        Return suffix if model_name exists in embedding_func, otherwise return None.
        Note: embedding_func is guaranteed to exist (validated in __post_init__).

        Returns:
            str | None: Suffix string e.g. "text_embedding_3_large_3072d", or None if model_name not available
        """
        import re

        # Check if model_name exists (model_name is optional in EmbeddingFunc)
        model_name = getattr(self.embedding_func, "model_name", None)
        if not model_name:
            return None

        # embedding_dim is required in EmbeddingFunc
        embedding_dim = self.embedding_func.embedding_dim

        # Generate suffix: clean model name and append dimension
        safe_model_name = re.sub(r"[^a-zA-Z0-9_]", "_", model_name.lower())
        return f"{safe_model_name}_{embedding_dim}d"

    # 向量搜索，返回前 top_k 个最相似的结果
    @abstractmethod
    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        """Query the vector storage and retrieve top_k results.

        Args:
            query: The query string to search for
            top_k: Number of top results to return
            query_embedding: Optional pre-computed embedding for the query.
                           If provided, skips embedding computation for better performance.
        """

    # 插入或更新向量
    @abstractmethod
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Insert or update vectors in the storage.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """

    # 删除单个实体
    @abstractmethod
    async def delete_entity(self, entity_name: str) -> None:
        """Delete a single entity by its name.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """

    # 删除实体关系
    @abstractmethod
    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete relations for a given entity.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """

    # 通过ID获取向量数据
    @abstractmethod
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        pass
    
    # 通过多个ID批量获取向量数据
    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        pass
    
    # 删除多个向量
    @abstractmethod
    async def delete(self, ids: list[str]):
        """Delete vectors with specified IDs

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            ids: List of vector IDs to be deleted
        """

    # 仅获取向量值（不含元数据），用于高效计算
    @abstractmethod
    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vectors by their IDs, returning only ID and vector data for efficiency

        Args:
            ids: List of unique identifiers

        Returns:
            Dictionary mapping IDs to their vector embeddings
            Format: {id: [vector_values], ...}
        """
        pass

# 用于存储文档、实体、关系等结构化数据
@dataclass
class BaseKVStorage(StorageNameSpace, ABC):
    embedding_func: EmbeddingFunc

    @abstractmethod
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get value by id"""

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get values by ids"""

    # 返回不存在的键
    @abstractmethod
    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return un-exist keys"""

    @abstractmethod
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Upsert data

        Importance notes for in-memory storage:
        1. 更改将在下次 index_done_callback 时保存到磁盘
        2. 更新标志以通知其他进程需要数据持久化
        1. Changes will be persisted to disk during the next index_done_callback
        2. update flags to notify other processes that data persistence is needed
        """

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete specific records from storage by their IDs

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. update flags to notify other processes that data persistence is needed

        Args:
            ids (list[str]): List of document IDs to be deleted from storage

        Returns:
            None
        """

    @abstractmethod
    async def is_empty(self) -> bool:
        """Check if the storage is empty

        Returns:
            bool: True if storage contains no data, False otherwise
        """

# 用于知识图谱的节点和边的存储 所有边都是无向的
@dataclass
class BaseGraphStorage(StorageNameSpace, ABC):
    """All operations related to edges in graph should be undirected."""

    embedding_func: EmbeddingFunc

    @abstractmethod
    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph.

        Args:
            node_id: The ID of the node to check

        Returns:
            True if the node exists, False otherwise
        """

    @abstractmethod
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes.

        Args:
            source_node_id: The ID of the source node
            target_node_id: The ID of the target node

        Returns:
            True if the edge exists, False otherwise
        """

    # 获取节点的度（连接的边数）
    @abstractmethod
    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of connected edges) of a node.

        Args:
            node_id: The ID of the node

        Returns:
            The number of edges connected to the node
        """

    # 获取边的度（源节点和目标节点的度数之和）
    @abstractmethod
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree of an edge (sum of degrees of its source and target nodes).

        Args:
            src_id: The ID of the source node
            tgt_id: The ID of the target node

        Returns:
            The sum of the degrees of the source and target nodes
        """

    @abstractmethod
    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its ID, returning only node properties.

        Args:
            node_id: The ID of the node to retrieve

        Returns:
            A dictionary of node properties if found, None otherwise
        """

    @abstractmethod
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """Get edge properties between two nodes.

        Args:
            source_node_id: The ID of the source node
            target_node_id: The ID of the target node

        Returns:
            A dictionary of edge properties if found, None otherwise
        """

    @abstractmethod
    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all edges connected to a node.

        Args:
            source_node_id: The ID of the node to get edges for

        Returns:
            A list of (source_id, target_id) tuples representing edges,
            or None if the node doesn't exist
        """
    # 批量操作
    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Get nodes as a batch using UNWIND

        Default implementation fetches nodes one by one.
        Override this method for better performance in storage backends
        that support batch operations.
        """
        result = {}
        for node_id in node_ids:
            node = await self.get_node(node_id)
            if node is not None:
                result[node_id] = node
        return result

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Node degrees as a batch using UNWIND

        Default implementation fetches node degrees one by one.
        Override this method for better performance in storage backends
        that support batch operations.
        """
        result = {}
        for node_id in node_ids:
            degree = await self.node_degree(node_id)
            result[node_id] = degree
        return result

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """Edge degrees as a batch using UNWIND also uses node_degrees_batch

        Default implementation calculates edge degrees one by one.
        Override this method for better performance in storage backends
        that support batch operations.
        """
        result = {}
        for src_id, tgt_id in edge_pairs:
            degree = await self.edge_degree(src_id, tgt_id)
            result[(src_id, tgt_id)] = degree
        return result

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        """Get edges as a batch using UNWIND

        Default implementation fetches edges one by one.
        Override this method for better performance in storage backends
        that support batch operations.
        """
        result = {}
        for pair in pairs:
            src_id = pair["src"]
            tgt_id = pair["tgt"]
            edge = await self.get_edge(src_id, tgt_id)
            if edge is not None:
                result[(src_id, tgt_id)] = edge
        return result

    # 批量获取某个节点连接的边
    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """Get nodes edges as a batch using UNWIND

        Default implementation fetches node edges one by one.
        Override this method for better performance in storage backends
        that support batch operations.
        """
        result = {}
        for node_id in node_ids:
            edges = await self.get_node_edges(node_id)
            result[node_id] = edges if edges is not None else []
        return result

    @abstractmethod
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Insert a new node or update an existing node in the graph.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            node_id: The ID of the node to insert or update
            node_data: A dictionary of node properties
        """

    @abstractmethod
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Insert a new edge or update an existing edge in the graph.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            source_node_id: The ID of the source node
            target_node_id: The ID of the target node
            edge_data: A dictionary of edge properties
        """

    @abstractmethod
    async def delete_node(self, node_id: str) -> None:
        """Delete a node from the graph.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            node_id: The ID of the node to delete
        """

    @abstractmethod
    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes

        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            nodes: List of node IDs to be deleted
        """

    @abstractmethod
    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """

    # TODO: deprecated
    @abstractmethod
    async def get_all_labels(self) -> list[str]:
        """Get all labels in the graph.

        Returns:
            A list of all node labels in the graph, sorted alphabetically
        """

    @abstractmethod
    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int = 1000
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.

        Args:
            node_label: Label of the starting node，* means all nodes
            max_depth: Maximum depth of the subgraph, Defaults to 3
            max_nodes: Maxiumu nodes to return, Defaults to 1000（BFS if possible)

        Returns:
            KnowledgeGraph object containing nodes and edges, with an is_truncated flag
            indicating whether the graph was truncated due to max_nodes limit
        """

    @abstractmethod
    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes in the graph.

        Returns:
            A list of all nodes, where each node is a dictionary of its properties
            (Edge is bidirectional for some storage implementation; deduplication must be handled by the caller)
        """

    @abstractmethod
    async def get_all_edges(self) -> list[dict]:
        """Get all edges in the graph.

        Returns:
            A list of all edges, where each edge is a dictionary of its properties
        """
    # 获取热门标签（按节点度数排序）
    @abstractmethod
    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Get popular labels by node degree (most connected entities)

        Args:
            limit: Maximum number of labels to return

        Returns:
            List of labels sorted by degree (highest first)
        """
    # 模糊搜索标签
    @abstractmethod
    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Search labels with fuzzy matching

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching labels sorted by relevance
        """

# 文档状态管理
class DocStatus(str, Enum):
    """Document processing status"""

    PENDING = "pending"  # 待处理
    PROCESSING = "processing"  # 处理中
    PREPROCESSED = "preprocessed"  # 预处理完成
    PROCESSED = "processed"  # 处理完成
    FAILED = "failed"  # 处理失败

# 跟踪文档处理状态
@dataclass
class DocProcessingStatus:
    """Document processing status data structure"""
    # 前100个字符的内容摘要，用于预览
    content_summary: str
    """First 100 chars of document content, used for preview"""
    # 文档总长度
    content_length: int
    """Total length of document"""
    # 文档文件路径
    file_path: str
    """File path of the document"""
    # 当前处理状态
    status: DocStatus
    """Current processing status"""
    # 创建时间
    created_at: str
    """ISO format timestamp when document was created"""
    # 最后更新时间
    updated_at: str
    """ISO format timestamp when document was last updated"""
    # 跟踪ID，用于监控进度
    track_id: str | None = None
    """Tracking ID for monitoring progress"""
    # 分块数量
    chunks_count: int | None = None
    """Number of chunks after splitting, used for processing"""
    # 分块ID列表
    chunks_list: list[str] | None = field(default_factory=list)
    """List of chunk IDs associated with this document, used for deletion"""
    # 错误信息
    error_msg: str | None = None
    """Error message if failed"""
    # 额外元数据
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""
    # 内部字段：多模态处理是否完成
    multimodal_processed: bool | None = field(default=None, repr=False)
    """Internal field: indicates if multimodal processing is complete. Not shown in repr() but accessible for debugging."""

    def __post_init__(self):
        """
        Handle status conversion based on multimodal_processed field.

        Business rules:
        - If multimodal_processed is False and status is PROCESSED,
          then change status to PREPROCESSED
        - The multimodal_processed field is kept (with repr=False) for internal use and debugging
        """
        # Apply status conversion logic
        if self.multimodal_processed is not None:
            if (
                self.multimodal_processed is False
                and self.status == DocStatus.PROCESSED
            ):
                self.status = DocStatus.PREPROCESSED

# 文档状态存储抽象基类
@dataclass
class DocStatusStorage(BaseKVStorage, ABC):
    """Base class for document status storage"""

    @abstractmethod
    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""

    @abstractmethod
    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific status"""

    @abstractmethod
    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific track_id"""

    # 分页查询
    @abstractmethod
    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        """Get documents with pagination support

        Args:
            status_filter: Filter by document status, None for all statuses
            page: Page number (1-based)
            page_size: Number of documents per page (10-200)
            sort_field: Field to sort by ('created_at', 'updated_at', 'id')
            sort_direction: Sort direction ('asc' or 'desc')

        Returns:
            Tuple of (list of (doc_id, DocProcessingStatus) tuples, total_count)
        """

    @abstractmethod
    async def get_all_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status for all documents

        Returns:
            Dictionary mapping status names to counts
        """

    @abstractmethod
    async def get_doc_by_file_path(self, file_path: str) -> dict[str, Any] | None:
        """Get document by file path

        Args:
            file_path: The file path to search for

        Returns:
            dict[str, Any] | None: Document data if found, None otherwise
            Returns the same format as get_by_ids method
        """


class StoragesStatus(str, Enum):
    """Storages status"""

    NOT_CREATED = "not_created"
    CREATED = "created"
    INITIALIZED = "initialized"
    FINALIZED = "finalized"


@dataclass
class DeletionResult:
    """Represents the result of a deletion operation."""

    status: Literal["success", "not_found", "fail"]
    doc_id: str
    message: str
    status_code: int = 200
    file_path: str | None = None


# Unified Query Result Data Structures for Reference List Support

# 统一查询结果
@dataclass
class QueryResult:
    """
    统一查询结果数据结构，适用于所有查询模式。

    Attributes:
        content: 非流式响应的文本内容
        response_iterator: 流式响应的异步迭代器
        raw_data: 完整的结构化数据，包括引用和元数据
        is_streaming: 是否为流式结果
    """
    # 非流式响应的文本内容
    content: Optional[str] = None
    # AsyncIterator[str] 是 Python 异步编程中的异步迭代器类型，用于表示可以逐步产生字符串值的异步数据流。
    # 流式响应的异步迭代器
    response_iterator: Optional[AsyncIterator[str]] = None
    raw_data: Optional[Dict[str, Any]] = None
    is_streaming: bool = False

    # 从 raw_data 中提取参考文献列表的便捷属性
    @property
    def reference_list(self) -> List[Dict[str, str]]:
        """
        Convenient property to extract reference list from raw_data.

        Returns:
            List[Dict[str, str]]: Reference list in format:
            [{"reference_id": "1", "file_path": "/path/to/file.pdf"}, ...]
        """
        if self.raw_data:
            return self.raw_data.get("data", {}).get("references", [])
        return []
    # 从 raw_data 中提取元数据的便捷属性
    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Convenient property to extract metadata from raw_data.

        Returns:
            Dict[str, Any]: Query metadata including query_mode, keywords, etc.
        """
        if self.raw_data:
            return self.raw_data.get("metadata", {})
        return {}

# 上下文结果
@dataclass
class QueryContextResult:
    """
    Unified query context result data structure.

    Attributes:
        context: LLM context string
        raw_data: Complete structured data including reference_list
    """
    # LLM 上下文字符串
    context: str
    # 包含参考文献的完整数据
    raw_data: Dict[str, Any]

    # 提取参考文献
    @property
    def reference_list(self) -> List[Dict[str, str]]:
        """Convenient property to extract reference list from raw_data."""
        return self.raw_data.get("data", {}).get("references", [])

# 定义了所有存储类型的抽象基类和核心数据结构：

# BaseKVStorage：键值存储接口
# BaseVectorStorage：向量存储接口
# BaseGraphStorage：图存储接口
# QueryParam：查询参数配置类
# StorageNameSpace：存储命名空间抽象类 