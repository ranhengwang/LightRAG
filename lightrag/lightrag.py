from __future__ import annotations

import traceback
import asyncio
import configparser
import inspect
import os
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    cast,
    final,
    Literal,
    Optional,
    List,
    Dict,
    Union,
)
from lightrag.prompt import PROMPTS
from lightrag.exceptions import PipelineCancelledException
from lightrag.constants import (
    DEFAULT_MAX_GLEANING,
    DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE,
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_TOP_K,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_RELATED_CHUNK_NUMBER,
    DEFAULT_KG_CHUNK_PICK_METHOD,
    DEFAULT_MIN_RERANK_SCORE,
    DEFAULT_SUMMARY_MAX_TOKENS,
    DEFAULT_SUMMARY_CONTEXT_SIZE,
    DEFAULT_SUMMARY_LENGTH_RECOMMENDED,
    DEFAULT_MAX_ASYNC,
    DEFAULT_MAX_PARALLEL_INSERT,
    DEFAULT_MAX_GRAPH_NODES,
    DEFAULT_MAX_SOURCE_IDS_PER_ENTITY,
    DEFAULT_MAX_SOURCE_IDS_PER_RELATION,
    DEFAULT_ENTITY_TYPES,
    DEFAULT_SUMMARY_LANGUAGE,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_EMBEDDING_TIMEOUT,
    DEFAULT_SOURCE_IDS_LIMIT_METHOD,
    DEFAULT_MAX_FILE_PATHS,
    DEFAULT_FILE_PATH_MORE_PLACEHOLDER,
)
from lightrag.utils import get_env_value

from lightrag.kg import (
    STORAGES,
    verify_storage_implementation,
)


from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_data_init_lock,
    get_default_workspace,
    set_default_workspace,
    get_namespace_lock,
)

from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
    QueryParam,
    StorageNameSpace,
    StoragesStatus,
    DeletionResult,
    OllamaServerInfos,
    QueryResult,
)
from lightrag.namespace import NameSpace
from lightrag.operate import (
    chunking_by_token_size,
    extract_entities,
    merge_nodes_and_edges,
    kg_query,
    naive_query,
    rebuild_knowledge_from_chunks,
)
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import (
    Tokenizer,
    TiktokenTokenizer,
    EmbeddingFunc,
    always_get_an_event_loop,
    compute_mdhash_id,
    lazy_external_import,
    priority_limit_async_func_call,
    get_content_summary,
    sanitize_text_for_encoding,
    check_storage_env_vars,
    generate_track_id,
    convert_to_user_format,
    logger,
    subtract_source_ids,
    make_relation_chunk_key,
    normalize_source_ids_limit_method,
)
from lightrag.types import KnowledgeGraph
from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

# TODO: TO REMOVE @Yannick
config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


@final
@dataclass
class LightRAG:
    """LightRAG: Simple and Fast Retrieval-Augmented Generation."""

    # Directory
    # ---
    # 使用 field - 适用于可变默认值或需要自定义行为，每个实例独立，  可以实现一个数据隔离的效果
    # 缓存和临时文件存储目录
    working_dir: str = field(default="./rag_storage")
    """Directory where cache and temporary files are stored."""

    # Storage
    # ---
    # 这里只是给了一个默认的，真正创建LightRAG实例时可以指定不同的存储后端
    # 传参来通过构造函数来创造实例
    # 键值存储
    kv_storage: str = field(default="JsonKVStorage")
    """Storage backend for key-value data."""
    # 向量存储
    vector_storage: str = field(default="NanoVectorDBStorage")
    """Storage backend for vector embeddings."""
    # 图存储
    graph_storage: str = field(default="NetworkXStorage")
    """Storage backend for knowledge graphs."""
    # 文件状态存储
    doc_status_storage: str = field(default="JsonDocStatusStorage")
    """Storage type for tracking document processing statuses."""

    # Workspace
    # ---
    # 工作区，用于数据隔离。如果未设置WORKSPACE环境变量，则默认为空字符串。
    workspace: str = field(default_factory=lambda: os.getenv("WORKSPACE", ""))
    """Workspace for data isolation. Defaults to empty string if WORKSPACE environment variable is not set."""

    # Logging (Deprecated, use setup_logger in utils.py instead)
    # ---
    log_level: int | None = field(default=None)
    log_file_path: str | None = field(default=None)

    # Query parameters
    # ---
    # 每个查询检索的实体/关系数量
    top_k: int = field(default=get_env_value("TOP_K", DEFAULT_TOP_K, int))
    """Number of entities/relations to retrieve for each query."""
    # 每个查询检索的文本块数量
    chunk_top_k: int = field(
        default=get_env_value("CHUNK_TOP_K", DEFAULT_CHUNK_TOP_K, int)
    )
    """Maximum number of chunks in context."""
    # 上下文中实体的最大token数量
    max_entity_tokens: int = field(
        default=get_env_value("MAX_ENTITY_TOKENS", DEFAULT_MAX_ENTITY_TOKENS, int)
    )
    """Maximum number of tokens for entity in context."""
    # 上下文中关系的最大token数量
    max_relation_tokens: int = field(
        default=get_env_value("MAX_RELATION_TOKENS", DEFAULT_MAX_RELATION_TOKENS, int)
    )
    """Maximum number of tokens for relation in context."""
    # 上下文中总的最大token数量，包括系统提示、实体、关系和文本块
    max_total_tokens: int = field(
        default=get_env_value("MAX_TOTAL_TOKENS", DEFAULT_MAX_TOTAL_TOKENS, int)
    )
    """Maximum total tokens in context (including system prompt, entities, relations and chunks)."""
    # 向量数据库检索的余弦相似度阈值
    cosine_threshold: int = field(
        default=get_env_value("COSINE_THRESHOLD", DEFAULT_COSINE_THRESHOLD, int)
    )
    """Cosine threshold of vector DB retrieval for entities, relations and chunks."""
    # 每个实体或关系获取的相关文本块数量
    related_chunk_number: int = field(
        default=get_env_value("RELATED_CHUNK_NUMBER", DEFAULT_RELATED_CHUNK_NUMBER, int)
    )
    """Number of related chunks to grab from single entity or relation."""
    # 知识图谱查询时选择文本块的方法, 'WEIGHT'表示基于权重选择，'VECTOR'表示基于嵌入相似度选择
    kg_chunk_pick_method: str = field(
        default=get_env_value("KG_CHUNK_PICK_METHOD", DEFAULT_KG_CHUNK_PICK_METHOD, str)
    )
    """Method for selecting text chunks: 'WEIGHT' for weight-based selection, 'VECTOR' for embedding similarity-based selection."""

    # Entity extraction
    # ---
    # 对于不明确的内容，实体提取的最大尝试次数
    entity_extract_max_gleaning: int = field(
        default=get_env_value("MAX_GLEANING", DEFAULT_MAX_GLEANING, int)
    )
    """Maximum number of entity extraction attempts for ambiguous content."""
    # 在合并实体/关系时强制使用LLM生成摘要的最小尝试次数
    force_llm_summary_on_merge: int = field(
        default=get_env_value(
            "FORCE_LLM_SUMMARY_ON_MERGE", DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE, int
        )
    )

    # Text chunking
    # ---
    # 文本块的最大token数量
    chunk_token_size: int = field(default=int(os.getenv("CHUNK_SIZE", 1200)))
    """Maximum number of tokens per text chunk when splitting documents."""
    # 文本块之间的重叠token数量
    chunk_overlap_token_size: int = field(
        default=int(os.getenv("CHUNK_OVERLAP_SIZE", 100))
    )
    """Number of overlapping tokens between consecutive text chunks to preserve context."""
    # 分词器实例
    tokenizer: Optional[Tokenizer] = field(default=None)
    """
    A function that returns a Tokenizer instance.
    If None, and a `tiktoken_model_name` is provided, a TiktokenTokenizer will be created.
    If both are None, the default TiktokenTokenizer is used.
    """
    # 用于tiktoken分词的模型名称
    tiktoken_model_name: str = field(default="gpt-4o-mini")
    """Model name used for tokenization when chunking text with tiktoken. Defaults to `gpt-4o-mini`."""

    chunking_func: Callable[
        # 下面这个List对应函数的参数类型
        [
            Tokenizer,
            str,
            Optional[str],
            bool,
            int,
            int,
        ],
        # 下面对应的是函数的返回值类型
        Union[List[Dict[str, Any]], Awaitable[List[Dict[str, Any]]]],
    ] = field(default_factory=lambda: chunking_by_token_size)
    # Awaitable[T] = 需要 await 才能得到 T 的对象
    """
    Custom chunking function for splitting text into chunks before processing.

    The function can be either synchronous or asynchronous.

    The function should take the following parameters:

        - `tokenizer`: A Tokenizer instance to use for tokenization.
        - `content`: The text to be split into chunks.
        - `split_by_character`: The character to split the text on. If None, the text is split into chunks of `chunk_token_size` tokens.
        - `split_by_character_only`: If True, the text is split only on the specified character.
        - `chunk_overlap_token_size`: The number of overlapping tokens between consecutive chunks.
        - `chunk_token_size`: The maximum number of tokens per chunk.


    The function should return a list of dictionaries (or an awaitable that resolves to a list),
    where each dictionary contains the following keys:
        - `tokens` (int): The number of tokens in the chunk.
        - `content` (str): The text content of the chunk.
        - `chunk_order_index` (int): Zero-based index indicating the chunk's order in the document.

    Defaults to `chunking_by_token_size` if not specified.
    """

    # Embedding
    # ---
    # 文本嵌入函数
    embedding_func: EmbeddingFunc | None = field(default=None)
    """Function for computing text embeddings. Must be set before use."""
    # 对于嵌入模型的token限制
    embedding_token_limit: int | None = field(default=None, init=False)
    """Token limit for embedding model. Set automatically from embedding_func.max_token_size in __post_init__."""
    # embedding计算的批处理大小
    embedding_batch_num: int = field(default=int(os.getenv("EMBEDDING_BATCH_NUM", 10)))
    """Batch size for embedding computations."""
    # embedding函数的最大并发调用数
    embedding_func_max_async: int = field(
        default=int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", 8))
    )
    """Maximum number of concurrent embedding function calls."""
        # 嵌入缓存配置
    embedding_cache_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    """Configuration for embedding cache.
    - enabled: If True, enables caching to avoid redundant computations.
    - similarity_threshold: Minimum similarity score to use cached embeddings.
    - use_llm_check: If True, validates cached embeddings using an LLM.
    """
    # 嵌入缓存的默认超时时间
    default_embedding_timeout: int = field(
        default=int(os.getenv("EMBEDDING_TIMEOUT", DEFAULT_EMBEDDING_TIMEOUT))
    )

    # LLM Configuration
    # ---
    #  大模型交互函数
    llm_model_func: Callable[..., object] | None = field(default=None)
    """Function for interacting with the large language model (LLM). Must be set before use."""
    # LLM模型名称
    llm_model_name: str = field(default="gpt-4o-mini")
    """Name of the LLM model used for generating responses."""
    # 实体/关系描述的最大token数量
    summary_max_tokens: int = field(
        default=int(os.getenv("SUMMARY_MAX_TOKENS", DEFAULT_SUMMARY_MAX_TOKENS))
    )
    """Maximum tokens allowed for entity/relation description."""
    # 每个LLM响应允许的最大token数量
    summary_context_size: int = field(
        default=int(os.getenv("SUMMARY_CONTEXT_SIZE", DEFAULT_SUMMARY_CONTEXT_SIZE))
    )
    """Maximum number of tokens allowed per LLM response."""
    # LLM摘要输出的推荐长度
    summary_length_recommended: int = field(
        default=int(
            os.getenv("SUMMARY_LENGTH_RECOMMENDED", DEFAULT_SUMMARY_LENGTH_RECOMMENDED)
        )
    )
    """Recommended length of LLM summary output."""
    # LLM模型的最大并发调用数
    llm_model_max_async: int = field(
        default=int(os.getenv("MAX_ASYNC", DEFAULT_MAX_ASYNC))
    )
    """Maximum number of concurrent LLM calls."""
    # LLM模型的其他关键字参数
    llm_model_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments passed to the LLM model function."""
    # LLM函数的默认超时时间
    default_llm_timeout: int = field(
        default=int(os.getenv("LLM_TIMEOUT", DEFAULT_LLM_TIMEOUT))
    )

    # Rerank Configuration
    # ---
    # 文档重排序函数
    rerank_model_func: Callable[..., object] | None = field(default=None)
    """Function for reranking retrieved documents. All rerank configurations (model name, API keys, top_k, etc.) should be included in this function. Optional."""
    # 重排序的最小分数阈值
    min_rerank_score: float = field(
        default=get_env_value("MIN_RERANK_SCORE", DEFAULT_MIN_RERANK_SCORE, float)
    )
    """Minimum rerank score threshold for filtering chunks after reranking."""

    # Storage
    # ---
    # 向量数据库存储的额外参数
    vector_db_storage_cls_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional parameters for vector database storage."""
    # LLM响应缓存启用标志
    enable_llm_cache: bool = field(default=True)
    """Enables caching for LLM responses to avoid redundant computations."""
    # 实体提取缓存启用标志
    enable_llm_cache_for_entity_extract: bool = field(default=True)
    """If True, enables caching for entity extraction steps to reduce LLM costs."""

    # Extensions
    # ---
    # 最大并行插入数
    max_parallel_insert: int = field(
        default=int(os.getenv("MAX_PARALLEL_INSERT", DEFAULT_MAX_PARALLEL_INSERT))
    )
    """Maximum number of parallel insert operations."""
    # 进行知识图谱查询时返回的最大节点数
    max_graph_nodes: int = field(
        default=get_env_value("MAX_GRAPH_NODES", DEFAULT_MAX_GRAPH_NODES, int)
    )
    """Maximum number of graph nodes to return in knowledge graph queries."""
    # 实体图谱和向量数据库中每个实体允许的最大源ID数量
    max_source_ids_per_entity: int = field(
        default=get_env_value(
            "MAX_SOURCE_IDS_PER_ENTITY", DEFAULT_MAX_SOURCE_IDS_PER_ENTITY, int
        )
    )
    """Maximum number of source (chunk) ids in entity Grpah + VDB."""
    # 每个关系图谱和向量数据库中允许的最大源ID数量
    max_source_ids_per_relation: int = field(
        default=get_env_value(
            "MAX_SOURCE_IDS_PER_RELATION",
            DEFAULT_MAX_SOURCE_IDS_PER_RELATION,
            int,
        )
    )
    """Maximum number of source (chunk) ids in relation Graph + VDB."""
    # 源ID限制策略
    source_ids_limit_method: str = field(
        default_factory=lambda: normalize_source_ids_limit_method(
            get_env_value(
                "SOURCE_IDS_LIMIT_METHOD",
                DEFAULT_SOURCE_IDS_LIMIT_METHOD,
                str,
            )
        )
    )
    """Strategy for enforcing source_id limits: IGNORE_NEW or FIFO."""

    # 实体/关系文件路径字段中允许存储的最大文件路径数量
    max_file_paths: int = field(
        default=get_env_value("MAX_FILE_PATHS", DEFAULT_MAX_FILE_PATHS, int)
    )
    """Maximum number of file paths to store in entity/relation file_path field."""

    # 文件路径超过最大数量时的占位符文本
    file_path_more_placeholder: str = field(default=DEFAULT_FILE_PATH_MORE_PLACEHOLDER)
    """Placeholder text when file paths exceed max_file_paths limit."""

    addon_params: dict[str, Any] = field(
        default_factory=lambda: {
            "language": get_env_value(
                "SUMMARY_LANGUAGE", DEFAULT_SUMMARY_LANGUAGE, str
            ),
            "entity_types": get_env_value("ENTITY_TYPES", DEFAULT_ENTITY_TYPES, list),
        }
    )

    # Storages Management
    # ---

    # TODO: Deprecated (LightRAG will never initialize storage automatically on creation，and finalize should be call before destroying)
    auto_manage_storages_states: bool = field(default=False)
    """If True, lightrag will automatically calls initialize_storages and finalize_storages at the appropriate times."""

    cosine_better_than_threshold: float = field(
        default=float(os.getenv("COSINE_THRESHOLD", 0.2))
    )
    """Threshold for cosine similarity to consider one vector better than another."""
    ollama_server_infos: Optional[OllamaServerInfos] = field(default=None)
    """Configuration for Ollama server information."""

    _storages_status: StoragesStatus = field(default=StoragesStatus.NOT_CREATED)

    def __post_init__(self):
        from lightrag.kg.shared_storage import (
            # 各种共享存储和锁机制的初始化
            initialize_share_data,
        )

        # Handle deprecated parameters
        if self.log_level is not None:
            warnings.warn(
                "WARNING: log_level parameter is deprecated, use setup_logger in utils.py instead",
                UserWarning,
                stacklevel=2,
            )
        if self.log_file_path is not None:
            warnings.warn(
                "WARNING: log_file_path parameter is deprecated, use setup_logger in utils.py instead",
                UserWarning,
                stacklevel=2,
            )

        # Remove these attributes to prevent their use
        if hasattr(self, "log_level"):
            delattr(self, "log_level")
        if hasattr(self, "log_file_path"):
            delattr(self, "log_file_path")

        # 初始化共享数据
        initialize_share_data()

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        # Verify storage implementation compatibility and environment variables
        storage_configs = [
            ("KV_STORAGE", self.kv_storage),
            ("VECTOR_STORAGE", self.vector_storage),
            ("GRAPH_STORAGE", self.graph_storage),
            ("DOC_STATUS_STORAGE", self.doc_status_storage),
        ]

        for storage_type, storage_name in storage_configs:
            # Verify storage implementation compatibility
            verify_storage_implementation(storage_type, storage_name)
            # Check environment variables
            check_storage_env_vars(storage_name)

        # Ensure vector_db_storage_cls_kwargs has required fields
        self.vector_db_storage_cls_kwargs = {
            "cosine_better_than_threshold": self.cosine_better_than_threshold,
            **self.vector_db_storage_cls_kwargs,
        }

        # Init Tokenizer
        # Post-initialization hook to handle backward compatabile tokenizer initialization based on provided parameters
        if self.tokenizer is None:
            if self.tiktoken_model_name:
                self.tokenizer = TiktokenTokenizer(self.tiktoken_model_name)
            else:
                self.tokenizer = TiktokenTokenizer()

        # Initialize ollama_server_infos if not provided
        if self.ollama_server_infos is None:
            self.ollama_server_infos = OllamaServerInfos()

        # Validate config
        if self.force_llm_summary_on_merge < 3:
            logger.warning(
                f"force_llm_summary_on_merge should be at least 3, got {self.force_llm_summary_on_merge}"
            )
        if self.summary_context_size > self.max_total_tokens:
            logger.warning(
                f"summary_context_size({self.summary_context_size}) should no greater than max_total_tokens({self.max_total_tokens})"
            )
        if self.summary_length_recommended > self.summary_max_tokens:
            logger.warning(
                f"max_total_tokens({self.summary_max_tokens}) should greater than summary_length_recommended({self.summary_length_recommended})"
            )

        # Init Embedding
        # Step 1: Capture embedding_func and max_token_size before applying rate_limit decorator
        original_embedding_func = self.embedding_func
        embedding_max_token_size = None
        if self.embedding_func and hasattr(self.embedding_func, "max_token_size"):
            embedding_max_token_size = self.embedding_func.max_token_size
            logger.debug(
                f"Captured embedding max_token_size: {embedding_max_token_size}"
            )
        self.embedding_token_limit = embedding_max_token_size

        # Fix global_config now
        # 将 LightRAG 实例转换为字典格式，专门用于将 @dataclass 装饰的类实例转换为字典
        global_config = asdict(self)
        # Restore original EmbeddingFunc object (asdict converts it to dict)
        global_config["embedding_func"] = original_embedding_func

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in global_config.items()])
        logger.debug(f"LightRAG init with param:\n  {_print_config}\n")

        # Step 2: Apply priority wrapper decorator to EmbeddingFunc's inner func
        # Create a NEW EmbeddingFunc instance with the wrapped func to avoid mutating the caller's object
        # This ensures _generate_collection_suffix can still access attributes (model_name, embedding_dim)
        # while preventing side effects when the same EmbeddingFunc is reused across multiple LightRAG instances
        if self.embedding_func is not None:
            wrapped_func = priority_limit_async_func_call(
                self.embedding_func_max_async,
                llm_timeout=self.default_embedding_timeout,
                queue_name="Embedding func",
            )(self.embedding_func.func)
            # Use dataclasses.replace() to create a new instance, leaving the original unchanged
            self.embedding_func = replace(self.embedding_func, func=wrapped_func)

        # Initialize all storages
        self.key_string_value_json_storage_cls: type[BaseKVStorage] = (
            self._get_storage_class(self.kv_storage)
        )  # type: ignore
        self.vector_db_storage_cls: type[BaseVectorStorage] = self._get_storage_class(
            self.vector_storage
        )  # type: ignore
        self.graph_storage_cls: type[BaseGraphStorage] = self._get_storage_class(
            self.graph_storage
        )  # type: ignore
        self.key_string_value_json_storage_cls = partial(  # type: ignore
            self.key_string_value_json_storage_cls, global_config=global_config
        )
        # partial函数将原函数再封装一层，固定住部分参数
        self.vector_db_storage_cls = partial(  # type: ignore
            self.vector_db_storage_cls, global_config=global_config
        )
        self.graph_storage_cls = partial(  # type: ignore
            self.graph_storage_cls, global_config=global_config
        )

        # Initialize document status storage
        self.doc_status_storage_cls = self._get_storage_class(self.doc_status_storage)

        self.llm_response_cache: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
            workspace=self.workspace,
            global_config=global_config,
            embedding_func=self.embedding_func,
        )

        self.text_chunks: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        self.full_docs: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_FULL_DOCS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        self.full_entities: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_FULL_ENTITIES,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        self.full_relations: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_FULL_RELATIONS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        self.entity_chunks: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_ENTITY_CHUNKS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        self.relation_chunks: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_RELATION_CHUNKS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        self.chunk_entity_relation_graph: BaseGraphStorage = self.graph_storage_cls(  # type: ignore
            namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        self.entities_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=NameSpace.VECTOR_STORE_ENTITIES,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
            meta_fields={"entity_name", "source_id", "content", "file_path"},
        )
        self.relationships_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=NameSpace.VECTOR_STORE_RELATIONSHIPS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id", "source_id", "content", "file_path"},
        )
        self.chunks_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=NameSpace.VECTOR_STORE_CHUNKS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
            meta_fields={"full_doc_id", "content", "file_path"},
        )

        # Initialize document status storage
        self.doc_status: DocStatusStorage = self.doc_status_storage_cls(
            namespace=NameSpace.DOC_STATUS,
            workspace=self.workspace,
            global_config=global_config,
            embedding_func=None,
        )

        # Directly use llm_response_cache, don't create a new object
        hashing_kv = self.llm_response_cache

        # Get timeout from LLM model kwargs for dynamic timeout calculation
        self.llm_model_func = priority_limit_async_func_call(
            self.llm_model_max_async,
            llm_timeout=self.default_llm_timeout,
            queue_name="LLM func",
        )(
            partial(
                self.llm_model_func,  # type: ignore
                hashing_kv=hashing_kv,
                **self.llm_model_kwargs,
            )
        )

        self._storages_status = StoragesStatus.CREATED

    async def initialize_storages(self):
        """Storage initialization must be called one by one to prevent deadlock"""
        if self._storages_status == StoragesStatus.CREATED:
            # Set the first initialized workspace will set the default workspace
            # Allows namespace operation without specifying workspace for backward compatibility
            default_workspace = get_default_workspace()
            if default_workspace is None:
                set_default_workspace(self.workspace)
            elif default_workspace != self.workspace:
                logger.info(
                    f"Creating LightRAG instance with workspace='{self.workspace}' "
                    f"while default workspace is set to '{default_workspace}'"
                )

            # Auto-initialize pipeline_status for this workspace
            from lightrag.kg.shared_storage import initialize_pipeline_status

            await initialize_pipeline_status(workspace=self.workspace)

            for storage in (
                self.full_docs,
                self.text_chunks,
                self.full_entities,
                self.full_relations,
                self.entity_chunks,
                self.relation_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
                self.llm_response_cache,
                self.doc_status,
            ):
                if storage:
                    # logger.debug(f"Initializing storage: {storage}")
                    await storage.initialize()

            self._storages_status = StoragesStatus.INITIALIZED
            logger.debug("All storage types initialized")

    async def finalize_storages(self):
        """
        异步关闭和清理所有存储组件，确保数据持久化、释放资源（如数据库连接、文件句柄），并妥善处理关闭过程中的错误
        Asynchronously finalize the storages with improved error handling"""
        if self._storages_status == StoragesStatus.INITIALIZED:
            storages = [
                ("full_docs", self.full_docs),
                ("text_chunks", self.text_chunks),
                ("full_entities", self.full_entities),
                ("full_relations", self.full_relations),
                ("entity_chunks", self.entity_chunks),
                ("relation_chunks", self.relation_chunks),
                ("entities_vdb", self.entities_vdb),
                ("relationships_vdb", self.relationships_vdb),
                ("chunks_vdb", self.chunks_vdb),
                ("chunk_entity_relation_graph", self.chunk_entity_relation_graph),
                ("llm_response_cache", self.llm_response_cache),
                ("doc_status", self.doc_status),
            ]

            # Finalize each storage individually to ensure one failure doesn't prevent others from closing
            # 记录成功关闭与失败关闭
            successful_finalizations = []
            failed_finalizations = []

            # 逐个关闭存储
            for storage_name, storage in storages:
                if storage:
                    try:
                        # 异步关闭存储组件，确保数据持久化和资源释放
                        await storage.finalize()
                        successful_finalizations.append(storage_name)
                        logger.debug(f"Successfully finalized {storage_name}")
                    except Exception as e:
                        error_msg = f"Failed to finalize {storage_name}: {e}"
                        logger.error(error_msg)
                        failed_finalizations.append(storage_name)

            # Log summary of finalization results
            if successful_finalizations:
                logger.info(
                    f"Successfully finalized {len(successful_finalizations)} storages"
                )

            if failed_finalizations:
                logger.error(
                    f"Failed to finalize {len(failed_finalizations)} storages: {', '.join(failed_finalizations)}"
                )
            else:
                logger.debug("All storages finalized successfully")

            self._storages_status = StoragesStatus.FINALIZED

    async def check_and_migrate_data(self):
        """检查并执行数据迁移，确保旧版本 LightRAG 的数据能够升级到新的存储结构。它是一个向后兼容的数据迁移工具。
        Check if data migration is needed and perform migration if necessary"""
        # 使用一个全局锁来防止多个实例同时进行迁移
        async with get_data_init_lock():
            try:
                # Check if migration is needed:
                # 1. chunk_entity_relation_graph has entities and relations (count > 0)
                # 2. full_entities and full_relations are empty

                # Get all entity labels from graph
                # # 获取图数据库中的所有实体标签
                all_entity_labels = (
                    await self.chunk_entity_relation_graph.get_all_labels()
                )

                if not all_entity_labels:
                    logger.debug("No entities found in graph, skipping migration check")
                    return

                try:
                    # Initialize chunk tracking storage after migration
                    # 迁移实体和关系与 chunk 的关联关系
                    await self._migrate_chunk_tracking_storage()
                except Exception as e:
                    logger.error(f"Error during chunk_tracking migration: {e}")
                    raise e

                # Check if full_entities and full_relations are empty
                # Get all processed documents to check their entity/relation data
                # 获取已处理文档列表
                try:
                    processed_docs = await self.doc_status.get_docs_by_status(
                        DocStatus.PROCESSED
                    )

                    if not processed_docs:
                        logger.debug("No processed documents found, skipping migration")
                        return

                    # Check first few documents to see if they have full_entities/full_relations data
                    # 检查是否需要迁移实体/关系数据
                    migration_needed = True
                    checked_count = 0
                    max_check = min(5, len(processed_docs))  # Check up to 5 documents

                    for doc_id in list(processed_docs.keys())[:max_check]:
                        checked_count += 1
                        entity_data = await self.full_entities.get_by_id(doc_id)
                        relation_data = await self.full_relations.get_by_id(doc_id)

                        if entity_data or relation_data:
                            migration_needed = False
                            break

                    if not migration_needed:
                        logger.debug(
                            "Full entities/relations data already exists, no migration needed"
                        )
                        return

                    logger.info(
                        f"Data migration needed: found {len(all_entity_labels)} entities in graph but no full_entities/full_relations data"
                    )

                    # Perform migration
                    # 执行实体和关系数据迁移
                    await self._migrate_entity_relation_data(processed_docs)

                except Exception as e:
                    logger.error(f"Error during migration check: {e}")
                    raise e

            except Exception as e:
                logger.error(f"Error in data migration check: {e}")
                raise e

    async def _migrate_entity_relation_data(self, processed_docs: dict):
        """Migrate existing entity and relation data to full_entities and full_relations storage"""
        logger.info(f"Starting data migration for {len(processed_docs)} documents")

        # Create mapping from chunk_id to doc_id
        chunk_to_doc = {}
        for doc_id, doc_status in processed_docs.items():
            chunk_ids = (
                doc_status.chunks_list
                if hasattr(doc_status, "chunks_list") and doc_status.chunks_list
                else []
            )
            for chunk_id in chunk_ids:
                chunk_to_doc[chunk_id] = doc_id

        # Initialize document entity and relation mappings
        doc_entities = {}  # doc_id -> set of entity_names
        doc_relations = {}  # doc_id -> set of relation_pairs (as tuples)

        # Get all nodes and edges from graph
        all_nodes = await self.chunk_entity_relation_graph.get_all_nodes()
        all_edges = await self.chunk_entity_relation_graph.get_all_edges()

        # Process all nodes once
        for node in all_nodes:
            if "source_id" in node:
                entity_id = node.get("entity_id") or node.get("id")
                if not entity_id:
                    continue

                # Get chunk IDs from source_id
                source_ids = node["source_id"].split(GRAPH_FIELD_SEP)

                # Find which documents this entity belongs to
                for chunk_id in source_ids:
                    doc_id = chunk_to_doc.get(chunk_id)
                    if doc_id:
                        if doc_id not in doc_entities:
                            doc_entities[doc_id] = set()
                        doc_entities[doc_id].add(entity_id)

        # Process all edges once
        for edge in all_edges:
            if "source_id" in edge:
                src = edge.get("source")
                tgt = edge.get("target")
                if not src or not tgt:
                    continue

                # Get chunk IDs from source_id
                source_ids = edge["source_id"].split(GRAPH_FIELD_SEP)

                # Find which documents this relation belongs to
                for chunk_id in source_ids:
                    doc_id = chunk_to_doc.get(chunk_id)
                    if doc_id:
                        if doc_id not in doc_relations:
                            doc_relations[doc_id] = set()
                        # Use tuple for set operations, convert to list later
                        doc_relations[doc_id].add(tuple(sorted((src, tgt))))

        # Store the results in full_entities and full_relations
        migration_count = 0

        # Store entities
        if doc_entities:
            entities_data = {}
            for doc_id, entity_set in doc_entities.items():
                entities_data[doc_id] = {
                    "entity_names": list(entity_set),
                    "count": len(entity_set),
                }
            await self.full_entities.upsert(entities_data)

        # Store relations
        if doc_relations:
            relations_data = {}
            for doc_id, relation_set in doc_relations.items():
                # Convert tuples back to lists
                relations_data[doc_id] = {
                    "relation_pairs": [list(pair) for pair in relation_set],
                    "count": len(relation_set),
                }
            await self.full_relations.upsert(relations_data)

        migration_count = len(
            set(list(doc_entities.keys()) + list(doc_relations.keys()))
        )

        # Persist the migrated data
        await self.full_entities.index_done_callback()
        await self.full_relations.index_done_callback()

        logger.info(
            f"Data migration completed: migrated {migration_count} documents with entities/relations"
        )

    async def _migrate_chunk_tracking_storage(self) -> None:
        """Ensure entity/relation chunk tracking KV stores exist and are seeded."""

        if not self.entity_chunks or not self.relation_chunks:
            return

        need_entity_migration = False
        need_relation_migration = False

        try:
            need_entity_migration = await self.entity_chunks.is_empty()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"Failed to check entity chunks storage: {exc}")
            raise exc

        try:
            need_relation_migration = await self.relation_chunks.is_empty()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"Failed to check relation chunks storage: {exc}")
            raise exc

        if not need_entity_migration and not need_relation_migration:
            return

        BATCH_SIZE = 500  # Process 500 records per batch

        if need_entity_migration:
            try:
                nodes = await self.chunk_entity_relation_graph.get_all_nodes()
            except Exception as exc:
                logger.error(f"Failed to fetch nodes for chunk migration: {exc}")
                nodes = []

            logger.info(f"Starting chunk_tracking data migration: {len(nodes)} nodes")

            # Process nodes in batches
            total_nodes = len(nodes)
            total_batches = (total_nodes + BATCH_SIZE - 1) // BATCH_SIZE
            total_migrated = 0

            for batch_idx in range(total_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min((batch_idx + 1) * BATCH_SIZE, total_nodes)
                batch_nodes = nodes[start_idx:end_idx]

                upsert_payload: dict[str, dict[str, object]] = {}
                for node in batch_nodes:
                    entity_id = node.get("entity_id") or node.get("id")
                    if not entity_id:
                        continue

                    raw_source = node.get("source_id") or ""
                    chunk_ids = [
                        chunk_id
                        for chunk_id in raw_source.split(GRAPH_FIELD_SEP)
                        if chunk_id
                    ]
                    if not chunk_ids:
                        continue

                    upsert_payload[entity_id] = {
                        "chunk_ids": chunk_ids,
                        "count": len(chunk_ids),
                    }

                if upsert_payload:
                    await self.entity_chunks.upsert(upsert_payload)
                    total_migrated += len(upsert_payload)
                    logger.info(
                        f"Processed entity batch {batch_idx + 1}/{total_batches}: {len(upsert_payload)} records (total: {total_migrated}/{total_nodes})"
                    )

            if total_migrated > 0:
                # Persist entity_chunks data to disk
                await self.entity_chunks.index_done_callback()
                logger.info(
                    f"Entity chunk_tracking migration completed: {total_migrated} records persisted"
                )

        if need_relation_migration:
            try:
                edges = await self.chunk_entity_relation_graph.get_all_edges()
            except Exception as exc:
                logger.error(f"Failed to fetch edges for chunk migration: {exc}")
                edges = []

            logger.info(f"Starting chunk_tracking data migration: {len(edges)} edges")

            # Process edges in batches
            total_edges = len(edges)
            total_batches = (total_edges + BATCH_SIZE - 1) // BATCH_SIZE
            total_migrated = 0

            for batch_idx in range(total_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min((batch_idx + 1) * BATCH_SIZE, total_edges)
                batch_edges = edges[start_idx:end_idx]

                upsert_payload: dict[str, dict[str, object]] = {}
                for edge in batch_edges:
                    src = edge.get("source") or edge.get("src_id") or edge.get("src")
                    tgt = edge.get("target") or edge.get("tgt_id") or edge.get("tgt")
                    if not src or not tgt:
                        continue

                    raw_source = edge.get("source_id") or ""
                    chunk_ids = [
                        chunk_id
                        for chunk_id in raw_source.split(GRAPH_FIELD_SEP)
                        if chunk_id
                    ]
                    if not chunk_ids:
                        continue

                    storage_key = make_relation_chunk_key(src, tgt)
                    upsert_payload[storage_key] = {
                        "chunk_ids": chunk_ids,
                        "count": len(chunk_ids),
                    }

                if upsert_payload:
                    await self.relation_chunks.upsert(upsert_payload)
                    total_migrated += len(upsert_payload)
                    logger.info(
                        f"Processed relation batch {batch_idx + 1}/{total_batches}: {len(upsert_payload)} records (total: {total_migrated}/{total_edges})"
                    )

            if total_migrated > 0:
                # Persist relation_chunks data to disk
                await self.relation_chunks.index_done_callback()
                logger.info(
                    f"Relation chunk_tracking migration completed: {total_migrated} records persisted"
                )

    async def get_graph_labels(self):
        """
        获取所有图谱标签
        """
        text = await self.chunk_entity_relation_graph.get_all_labels()
        return text

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = None,
    ) -> KnowledgeGraph:
        """Get knowledge graph for a given label，这个知识图谱包括节点和边

        Args:
            node_label (str): Label to get knowledge graph for
            max_depth (int): Maximum depth of graph  子图的跳数
            max_nodes (int, optional): Maximum number of nodes to return. Defaults to self.max_graph_nodes.

        Returns:
            KnowledgeGraph: Knowledge graph containing nodes and edges
        """
        # Use self.max_graph_nodes as default if max_nodes is None
        if max_nodes is None:
            max_nodes = self.max_graph_nodes
        else:
            # Limit max_nodes to not exceed self.max_graph_nodes
            max_nodes = min(max_nodes, self.max_graph_nodes)

        return await self.chunk_entity_relation_graph.get_knowledge_graph(
            node_label, max_depth, max_nodes
        )

    def _get_storage_class(self, storage_name: str) -> Callable[..., Any]:
        """
        需要哪个就引入哪个，获取输入storage_name对应的存储类
        """
        # Direct imports for default storage implementations
        if storage_name == "JsonKVStorage":
            from lightrag.kg.json_kv_impl import JsonKVStorage

            return JsonKVStorage
        elif storage_name == "NanoVectorDBStorage":
            from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage

            return NanoVectorDBStorage
        elif storage_name == "NetworkXStorage":
            from lightrag.kg.networkx_impl import NetworkXStorage

            return NetworkXStorage
        elif storage_name == "JsonDocStatusStorage":
            from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage

            return JsonDocStatusStorage
        else:
            # Fallback to dynamic import for other storage implementations
            import_path = STORAGES[storage_name]
            storage_class = lazy_external_import(import_path, storage_name)
            return storage_class

    def insert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
        track_id: str | None = None,
    ) -> str:
        """Sync Insert documents with checkpoint support

        Args:
            input: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            chunk_token_size, it will be split again by token size.
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
            ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: single string of the file path or list of file paths, used for citation
            track_id: tracking ID for monitoring processing status, if not provided, will be generated

        Returns:
            str: tracking ID for monitoring processing status
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.ainsert(
                input,
                split_by_character,
                split_by_character_only,
                ids,
                file_paths,
                track_id,
            )
        )

    async def ainsert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
        track_id: str | None = None,
    ) -> str:
        """Async Insert documents with checkpoint support

        Args:
            input: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            chunk_token_size, it will be split again by token size.
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
            ids: list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: list of file paths corresponding to each document, used for citation
            track_id: tracking ID for monitoring processing status, if not provided, will be generated

        Returns:
            str: tracking ID for monitoring processing status
        """
        # Generate track_id if not provided
        if track_id is None:
            track_id = generate_track_id("insert")

        await self.apipeline_enqueue_documents(input, ids, file_paths, track_id)
        await self.apipeline_process_enqueue_documents(
            split_by_character, split_by_character_only
        )

        return track_id

    # TODO: deprecated, use insert instead
    def insert_custom_chunks(
        self,
        full_text: str,
        text_chunks: list[str],
        doc_id: str | list[str] | None = None,
    ) -> None:
        loop = always_get_an_event_loop()
        loop.run_until_complete(
            self.ainsert_custom_chunks(full_text, text_chunks, doc_id)
        )

    # TODO: deprecated, use ainsert instead
    async def ainsert_custom_chunks(
        self, full_text: str, text_chunks: list[str], doc_id: str | None = None
    ) -> None:
        update_storage = False
        try:
            # Clean input texts
            full_text = sanitize_text_for_encoding(full_text)
            text_chunks = [sanitize_text_for_encoding(chunk) for chunk in text_chunks]
            file_path = ""

            # Process cleaned texts
            if doc_id is None:
                doc_key = compute_mdhash_id(full_text, prefix="doc-")
            else:
                doc_key = doc_id
            new_docs = {doc_key: {"content": full_text, "file_path": file_path}}

            _add_doc_keys = await self.full_docs.filter_keys({doc_key})
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("This document is already in the storage.")
                return

            update_storage = True
            logger.info(f"Inserting {len(new_docs)} docs")

            inserting_chunks: dict[str, Any] = {}
            for index, chunk_text in enumerate(text_chunks):
                chunk_key = compute_mdhash_id(chunk_text, prefix="chunk-")
                tokens = len(self.tokenizer.encode(chunk_text))
                inserting_chunks[chunk_key] = {
                    "content": chunk_text,
                    "full_doc_id": doc_key,
                    "tokens": tokens,
                    "chunk_order_index": index,
                    "file_path": file_path,
                }

            doc_ids = set(inserting_chunks.keys())
            add_chunk_keys = await self.text_chunks.filter_keys(doc_ids)
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage.")
                return

            tasks = [
                self.chunks_vdb.upsert(inserting_chunks),
                self._process_extract_entities(inserting_chunks),
                self.full_docs.upsert(new_docs),
                self.text_chunks.upsert(inserting_chunks),
            ]
            await asyncio.gather(*tasks)

        finally:
            if update_storage:
                await self._insert_done()

    async def apipeline_enqueue_documents(
        self,
        input: str | list[str],
        ids: list[str] | None = None,
        file_paths: str | list[str] | None = None,
        track_id: str | None = None,
    ) -> str:
        """
        文档处理流水线的入队阶段，负责接收文档并将其加入处理队列。
        它是文档索引的第一步，主要完成文档的验证、去重、状态初始化和持久化。
        Pipeline for Processing Documents

        1. Validate ids if provided or generate MD5 hash IDs and remove duplicate contents
        2. Generate document initial status
        3. Filter out already processed documents
        4. Enqueue document in status

        Args:
            input: Single document string or list of document strings
            ids: list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: list of file paths corresponding to each document, used for citation
            track_id: tracking ID for monitoring processing status, if not provided, will be generated with "enqueue" prefix

        Returns:
            str: tracking ID for monitoring processing status
        """
        # Generate track_id if not provided
        if track_id is None or track_id.strip() == "":
            track_id = generate_track_id("enqueue")
        if isinstance(input, str):
            input = [input]
        if isinstance(ids, str):
            ids = [ids]
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        # If file_paths is provided, ensure it matches the number of documents
        # 处理文件路径
        if file_paths is not None:
            if isinstance(file_paths, str):
                file_paths = [file_paths]
            if len(file_paths) != len(input):
                raise ValueError(
                    "Number of file paths must match the number of documents"
                )
        else:
            # If no file paths provided, use placeholder
            file_paths = ["unknown_source"] * len(input)

        # 1. Validate ids if provided or generate MD5 hash IDs and remove duplicate contents
        if ids is not None:
            # Check if the number of IDs matches the number of documents
            # 验证ID数量
            if len(ids) != len(input):
                raise ValueError("Number of IDs must match the number of documents")

            # Check if IDs are unique
            # 验证ID的唯一性
            if len(ids) != len(set(ids)):
                raise ValueError("IDs must be unique")

            # Generate contents dict and remove duplicates in one pass
            # 去重：相同内容只保留第一个
            unique_contents = {}
            for id_, doc, path in zip(ids, input, file_paths):
                cleaned_content = sanitize_text_for_encoding(doc)
                if cleaned_content not in unique_contents:
                    unique_contents[cleaned_content] = (id_, path)

            # Reconstruct contents with unique content
            # 重建 contents 字典
            contents = {
                id_: {"content": content, "file_path": file_path}
                for content, (id_, file_path) in unique_contents.items()
            }
        else:
            # Clean input text and remove duplicates in one pass
            # 清理文本并且去重
            unique_content_with_paths = {}
            for doc, path in zip(input, file_paths):
                cleaned_content = sanitize_text_for_encoding(doc)
                if cleaned_content not in unique_content_with_paths:
                    unique_content_with_paths[cleaned_content] = path

            # Generate contents dict of MD5 hash IDs and documents with paths
            # 生成 MD5 哈希作为 doc_id
            contents = {
                compute_mdhash_id(content, prefix="doc-"): {
                    "content": content,
                    "file_path": path,
                }
                for content, path in unique_content_with_paths.items()
            }

        # 2. Generate document initial status (without content)
        # 生成文档初始状态
        new_docs: dict[str, Any] = {
            id_: {
                "status": DocStatus.PENDING,
                "content_summary": get_content_summary(content_data["content"]),
                "content_length": len(content_data["content"]),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "file_path": content_data[
                    "file_path"
                ],  # Store file path in document status
                "track_id": track_id,  # Store track_id in document status
            }
            for id_, content_data in contents.items()
        }

        # 3. Filter out already processed documents,过滤已存在的文档
        # Get docs ids
        # 获取所有新文档 ID
        all_new_doc_ids = set(new_docs.keys())
        # Exclude IDs of documents that are already enqueued
        # 调用 doc_status 存储的 filter_keys，返回不存在的 ID
        unique_new_doc_ids = await self.doc_status.filter_keys(all_new_doc_ids)

        # Log ignored document IDs (documents that were filtered out because they already exist)
        # 记录被忽略的文档
        ignored_ids = list(all_new_doc_ids - unique_new_doc_ids)
        if ignored_ids:
            for doc_id in ignored_ids:
                file_path = new_docs.get(doc_id, {}).get("file_path", "unknown_source")
                logger.warning(
                    f"Ignoring document ID (already exists): {doc_id} ({file_path})"
                )
            if len(ignored_ids) > 3:
                logger.warning(
                    f"Total Ignoring {len(ignored_ids)} document IDs that already exist in storage"
                )

        # Filter new_docs to only include documents with unique IDs
        new_docs = {
            doc_id: new_docs[doc_id]
            for doc_id in unique_new_doc_ids
            if doc_id in new_docs
        }

        if not new_docs:
            logger.warning("No new unique documents were found.")
            return

        # 4. Store document content in full_docs and status in doc_status
        #    Store full document content separately
        # 准备完整的文档数据
        full_docs_data = {
            doc_id: {
                "content": contents[doc_id]["content"],
                "file_path": contents[doc_id]["file_path"],
            }
            for doc_id in new_docs.keys()
        }
        # 存储到 full_docs（完整文档内容）
        await self.full_docs.upsert(full_docs_data)
        # Persist data to disk immediately
        # 立即持久化到磁盘
        await self.full_docs.index_done_callback()

        # Store document status (without content)
        await self.doc_status.upsert(new_docs)
        logger.debug(f"Stored {len(new_docs)} new unique documents")

        return track_id

    async def apipeline_enqueue_error_documents(
        self,
        error_files: list[dict[str, Any]],
        track_id: str | None = None,
    ) -> None:
        """
        Record file extraction errors in doc_status storage.

        This function creates error document entries in the doc_status storage for files
        that failed during the extraction process. Each error entry contains information
        about the failure to help with debugging and monitoring.

        Args:
            error_files: List of dictionaries containing error information for each failed file.
                Each dictionary should contain:
                - file_path: Original file name/path
                - error_description: Brief error description (for content_summary)
                - original_error: Full error message (for error_msg)
                - file_size: File size in bytes (for content_length, 0 if unknown)
            track_id: Optional tracking ID for grouping related operations

        Returns:
            None
        """
        if not error_files:
            logger.debug("No error files to record")
            return

        # Generate track_id if not provided
        if track_id is None or track_id.strip() == "":
            track_id = generate_track_id("error")

        error_docs: dict[str, Any] = {}
        current_time = datetime.now(timezone.utc).isoformat()

        for error_file in error_files:
            file_path = error_file.get("file_path", "unknown_file")
            error_description = error_file.get(
                "error_description", "File extraction failed"
            )
            original_error = error_file.get("original_error", "Unknown error")
            file_size = error_file.get("file_size", 0)

            # Generate unique doc_id with "error-" prefix
            doc_id_content = f"{file_path}-{error_description}"
            doc_id = compute_mdhash_id(doc_id_content, prefix="error-")

            error_docs[doc_id] = {
                "status": DocStatus.FAILED,
                "content_summary": error_description,
                "content_length": file_size,
                "error_msg": original_error,
                "chunks_count": 0,  # No chunks for failed files
                "created_at": current_time,
                "updated_at": current_time,
                "file_path": file_path,
                "track_id": track_id,
                "metadata": {
                    "error_type": "file_extraction_error",
                },
            }

        # Store error documents in doc_status
        if error_docs:
            await self.doc_status.upsert(error_docs)
            # Log each error for debugging
            for doc_id, error_doc in error_docs.items():
                logger.error(
                    f"File processing error: - ID: {doc_id} {error_doc['file_path']}"
                )

    async def _validate_and_fix_document_consistency(
        self,
        to_process_docs: dict[str, DocProcessingStatus],    #待处理文档列表
        pipeline_status: dict,       # 管道状态（跨进程共享）
        pipeline_status_lock: asyncio.Lock,
    ) -> dict[str, DocProcessingStatus]:
        """验证并修复文档数据的一致性，确保 doc_status 存储中的记录与 full_docs 存储中的实际内容保持同步
        返回修复后的文档列表
        这个函数是 apipeline_process_enqueue_documents 处理流程的第一道防线，确保后续处理的文档都是数据完整且状态正确的。
        Validate and fix document data consistency by deleting inconsistent entries, but preserve failed documents"""
        # 不一致文档（需要删除）
        inconsistent_docs = []
        # 失败文档，需要保留
        failed_docs_to_preserve = []
        successful_deletions = 0

        # 失败文档（DocStatus.FAILED）：保留在 doc_status 中供人工排查
        # 其他不一致文档：删除 doc_status 中的"孤儿"记录

        # Check each document's data consistency
        for doc_id, status_doc in to_process_docs.items():
            # Check if corresponding content exists in full_docs
            content_data = await self.full_docs.get_by_id(doc_id)
            if not content_data:    # doc_status 有记录，但 full_docs 没内容
                # Check if this is a failed document that should be preserved
                if (
                    hasattr(status_doc, "status")
                    and status_doc.status == DocStatus.FAILED
                ):
                    # 保留失败记录
                    failed_docs_to_preserve.append(doc_id)
                else:
                    # 标记为不一致文档
                    inconsistent_docs.append(doc_id)

        # Log information about failed documents that will be preserved
        # 保留失败文档记录
        # 失败文档可能包含有价值的错误信息，不应自动删除
        if failed_docs_to_preserve:
            async with pipeline_status_lock:
                preserve_message = f"Preserving {len(failed_docs_to_preserve)} failed document entries for manual review"
                logger.info(preserve_message)
                pipeline_status["latest_message"] = preserve_message
                pipeline_status["history_messages"].append(preserve_message)

            # Remove failed documents from processing list but keep them in doc_status
            # 从待处理列表中移除（但不删除 doc_status 记录）
            for doc_id in failed_docs_to_preserve:
                to_process_docs.pop(doc_id, None)

        # Delete inconsistent document entries(excluding failed documents)
        # 删除不一致的文档记录
        if inconsistent_docs:
            async with pipeline_status_lock:
                summary_message = (
                    f"Inconsistent document entries found: {len(inconsistent_docs)}"
                )
                logger.info(summary_message)
                pipeline_status["latest_message"] = summary_message
                pipeline_status["history_messages"].append(summary_message)

            successful_deletions = 0
            for doc_id in inconsistent_docs:
                try:
                    status_doc = to_process_docs[doc_id]
                    file_path = getattr(status_doc, "file_path", "unknown_source")

                    # Delete doc_status entry
                    await self.doc_status.delete([doc_id])
                    successful_deletions += 1

                    # Log successful deletion
                    async with pipeline_status_lock:
                        log_message = (
                            f"Deleted inconsistent entry: {doc_id} ({file_path})"
                        )
                        logger.info(log_message)
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

                    # Remove from processing list
                    to_process_docs.pop(doc_id, None)

                except Exception as e:
                    # Log deletion failure
                    async with pipeline_status_lock:
                        error_message = f"Failed to delete entry: {doc_id} - {str(e)}"
                        logger.error(error_message)
                        pipeline_status["latest_message"] = error_message
                        pipeline_status["history_messages"].append(error_message)

        # Final summary log
        # async with pipeline_status_lock:
        #     final_message = f"Successfully deleted {successful_deletions} inconsistent entries, preserved {len(failed_docs_to_preserve)} failed documents"
        #     logger.info(final_message)
        #     pipeline_status["latest_message"] = final_message
        #     pipeline_status["history_messages"].append(final_message)

        # Reset PROCESSING and FAILED documents that pass consistency checks to PENDING status

        # 重置异常状态文档
        docs_to_reset = {}
        reset_count = 0

        for doc_id, status_doc in to_process_docs.items():
            # Check if document has corresponding content in full_docs (consistency check)
            content_data = await self.full_docs.get_by_id(doc_id)
            if content_data:  # Document passes consistency check
                # Check if document is in PROCESSING or FAILED status
                if hasattr(status_doc, "status") and status_doc.status in [
                    DocStatus.PROCESSING,
                    DocStatus.FAILED,
                ]:
                    # Prepare document for status reset to PENDING
                    docs_to_reset[doc_id] = {
                        "status": DocStatus.PENDING,
                        "content_summary": status_doc.content_summary,
                        "content_length": status_doc.content_length,
                        "created_at": status_doc.created_at,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "file_path": getattr(status_doc, "file_path", "unknown_source"),
                        "track_id": getattr(status_doc, "track_id", ""),
                        # Clear any error messages and processing metadata
                        "error_msg": "",
                        "metadata": {},
                    }

                    # Update the status in to_process_docs as well
                    status_doc.status = DocStatus.PENDING
                    reset_count += 1

        # Update doc_status storage if there are documents to reset
        if docs_to_reset:
            await self.doc_status.upsert(docs_to_reset)

            async with pipeline_status_lock:
                reset_message = f"Reset {reset_count} documents from PROCESSING/FAILED to PENDING status"
                logger.info(reset_message)
                pipeline_status["latest_message"] = reset_message
                pipeline_status["history_messages"].append(reset_message)

        return to_process_docs

    async def apipeline_process_enqueue_documents(
        self,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
    ) -> None:
        """
        负责将已经入队的文档进行分块、实体关系提取和知识图谱构建
        Process pending documents by splitting them into chunks, processing
        each chunk for entity and relation extraction, and updating the
        document status.

        1. Get all pending, failed, and abnormally terminated processing documents.
        2. Validate document data consistency and fix any issues
        3. Split document content into chunks
        4. Process each chunk for entity and relation extraction
        5. Update the document status
        """

        # Get pipeline status shared data and lock
        # 获取跨进程共享的管道状态
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=self.workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=self.workspace
        )

        # Check if another process is already processing the queue
        async with pipeline_status_lock:
            # Ensure only one worker is processing documents
            if not pipeline_status.get("busy", False):
                # 获取所有待处理文档（PROCESSING、FAILED、PENDING 状态）
                processing_docs, failed_docs, pending_docs = await asyncio.gather(
                    self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
                    self.doc_status.get_docs_by_status(DocStatus.FAILED),
                    self.doc_status.get_docs_by_status(DocStatus.PENDING),
                )

                to_process_docs: dict[str, DocProcessingStatus] = {}
                # 异常中断的处理中文档
                to_process_docs.update(processing_docs)
                # 失败文档
                to_process_docs.update(failed_docs)
                # 待处理文档
                to_process_docs.update(pending_docs)

                if not to_process_docs:
                    logger.info("No documents to process")
                    return

                # 多进程并发控制
                pipeline_status.update(
                    {
                        "busy": True,       #标记为忙碌，阻止其他进程
                        "job_name": "Default Job",
                        "job_start": datetime.now(timezone.utc).isoformat(),
                        "docs": 0,
                        "batchs": 0,  # Total number of files to be processed
                        "cur_batch": 0,  # Number of files already processed
                        "request_pending": False,  # Clear any previous request
                        "cancellation_requested": False,  # Initialize cancellation flag
                        "latest_message": "",
                    }
                )
                # Cleaning history_messages without breaking it as a shared list object
                del pipeline_status["history_messages"][:]
            else:
                # Another process is busy, just set request flag and return
                # 设置待处理标志
                pipeline_status["request_pending"] = True
                logger.info(
                    "Another process is already processing the document queue. Request queued."
                )
                return

        try:
            # Process documents until no more documents or requests
            while True:
                # Check for cancellation request at the start of main loop
                # 在主循环开始时检查取消请求
                async with pipeline_status_lock:
                    if pipeline_status.get("cancellation_requested", False):
                        # Clear pending request
                        pipeline_status["request_pending"] = False
                        # Clear cancellation flag
                        pipeline_status["cancellation_requested"] = False

                        log_message = "Pipeline cancelled by user"
                        logger.info(log_message)
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

                        # Exit directly, skipping request_pending check
                        return

                # 说明所有文档都被处理结束
                if not to_process_docs:
                    log_message = "All enqueued documents have been processed"
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)
                    break

                # Validate document data consistency and fix any issues as part of the pipeline
                # 检查文档状态与实际内容的一致性
                to_process_docs = await self._validate_and_fix_document_consistency(
                    to_process_docs, pipeline_status, pipeline_status_lock
                )

                if not to_process_docs:
                    log_message = (
                        "No valid documents to process after consistency check"
                    )
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)
                    break

                log_message = f"Processing {len(to_process_docs)} document(s)"
                logger.info(log_message)

                # Update pipeline_status, batchs now represents the total number of files to be processed
                pipeline_status["docs"] = len(to_process_docs)
                pipeline_status["batchs"] = len(to_process_docs)
                pipeline_status["cur_batch"] = 0
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

                # Get first document's file path and total count for job name
                first_doc_id, first_doc = next(iter(to_process_docs.items()))
                first_doc_path = first_doc.file_path

                # Handle cases where first_doc_path is None
                if first_doc_path:
                    path_prefix = first_doc_path[:20] + (
                        "..." if len(first_doc_path) > 20 else ""
                    )
                else:
                    path_prefix = "unknown_source"

                total_files = len(to_process_docs)
                job_name = f"{path_prefix}[{total_files} files]"
                pipeline_status["job_name"] = job_name

                # Create a counter to track the number of processed files
                processed_count = 0
                # Create a semaphore to limit the number of concurrent file processing
                semaphore = asyncio.Semaphore(self.max_parallel_insert)

                async def process_document(
                    doc_id: str,
                    status_doc: DocProcessingStatus,
                    split_by_character: str | None,
                    split_by_character_only: bool,
                    pipeline_status: dict,
                    pipeline_status_lock: asyncio.Lock,
                    semaphore: asyncio.Semaphore,
                ) -> None:
                    """Process single document
                    主要步骤包括分块、文本块和实体等的存储、实体关系提取、节点合并、状态更新等
                    """
                    # Initialize variables at the start to prevent UnboundLocalError in error handling
                    file_path = "unknown_source"
                    current_file_number = 0
                    file_extraction_stage_ok = False
                    processing_start_time = int(time.time())
                    first_stage_tasks = []
                    entity_relation_task = None

                    async with semaphore:
                        nonlocal processed_count
                        # Initialize to prevent UnboundLocalError in error handling
                        first_stage_tasks = []
                        entity_relation_task = None
                        try:
                            # Check for cancellation before starting document processing
                            async with pipeline_status_lock:
                                if pipeline_status.get("cancellation_requested", False):
                                    raise PipelineCancelledException("User cancelled")

                            # Get file path from status document
                            file_path = getattr(
                                status_doc, "file_path", "unknown_source"
                            )

                            async with pipeline_status_lock:
                                # Update processed file count and save current file number
                                processed_count += 1
                                current_file_number = (
                                    processed_count  # Save the current file number
                                )
                                pipeline_status["cur_batch"] = processed_count

                                log_message = f"Extracting stage {current_file_number}/{total_files}: {file_path}"
                                logger.info(log_message)
                                pipeline_status["history_messages"].append(log_message)
                                log_message = f"Processing d-id: {doc_id}"
                                logger.info(log_message)
                                pipeline_status["latest_message"] = log_message
                                pipeline_status["history_messages"].append(log_message)

                                # Prevent memory growth: keep only latest 5000 messages when exceeding 10000
                                if len(pipeline_status["history_messages"]) > 10000:
                                    logger.info(
                                        f"Trimming pipeline history from {len(pipeline_status['history_messages'])} to 5000 messages"
                                    )
                                    pipeline_status["history_messages"] = (
                                        pipeline_status["history_messages"][-5000:]
                                    )

                            # Get document content from full_docs
                            content_data = await self.full_docs.get_by_id(doc_id)
                            if not content_data:
                                raise Exception(
                                    f"Document content not found in full_docs for doc_id: {doc_id}"
                                )
                            content = content_data["content"]

                            # Call chunking function, supporting both sync and async implementations
                            chunking_result = self.chunking_func(
                                self.tokenizer,
                                content,
                                split_by_character,
                                split_by_character_only,
                                self.chunk_overlap_token_size,
                                self.chunk_token_size,
                            )

                            # If result is awaitable, await to get actual result
                            if inspect.isawaitable(chunking_result):
                                chunking_result = await chunking_result

                            # Validate return type
                            if not isinstance(chunking_result, (list, tuple)):
                                raise TypeError(
                                    f"chunking_func must return a list or tuple of dicts, "
                                    f"got {type(chunking_result)}"
                                )

                            # Build chunks dictionary
                            chunks: dict[str, Any] = {
                                compute_mdhash_id(dp["content"], prefix="chunk-"): {
                                    **dp,
                                    "full_doc_id": doc_id,
                                    "file_path": file_path,  # Add file path to each chunk
                                    "llm_cache_list": [],  # Initialize empty LLM cache list for each chunk
                                }
                                for dp in chunking_result
                            }

                            if not chunks:
                                logger.warning("No document chunks to process")

                            # Record processing start time
                            processing_start_time = int(time.time())

                            # Check for cancellation before entity extraction
                            async with pipeline_status_lock:
                                if pipeline_status.get("cancellation_requested", False):
                                    raise PipelineCancelledException("User cancelled")

                            # Process document in two stages
                            # 两阶段并行处理
                            # Stage 1: Process text chunks and docs (parallel execution)
                            # 阶段 1：文本块和文档状态存储（并行）
                            # 这里总共三个并行任务
                            doc_status_task = asyncio.create_task(
                                self.doc_status.upsert(
                                    {
                                        doc_id: {
                                            "status": DocStatus.PROCESSING,
                                            "chunks_count": len(chunks),
                                            "chunks_list": list(
                                                chunks.keys()
                                            ),  # Save chunks list
                                            "content_summary": status_doc.content_summary,
                                            "content_length": status_doc.content_length,
                                            "created_at": status_doc.created_at,
                                            "updated_at": datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                            "file_path": file_path,
                                            "track_id": status_doc.track_id,  # Preserve existing track_id
                                            "metadata": {
                                                "processing_start_time": processing_start_time
                                            },
                                        }
                                    }
                                )
                            )
                            # 存储到向量数据库
                            chunks_vdb_task = asyncio.create_task(
                                self.chunks_vdb.upsert(chunks)
                            )
                            # 存储文本块
                            text_chunks_task = asyncio.create_task(
                                self.text_chunks.upsert(chunks)
                            )

                            # First stage tasks (parallel execution)
                            first_stage_tasks = [
                                doc_status_task,
                                chunks_vdb_task,
                                text_chunks_task,
                            ]
                            entity_relation_task = None

                            # Execute first stage tasks
                            # 并行执行第一阶段的几个任务
                            await asyncio.gather(*first_stage_tasks)

                            # Stage 2: Process entity relation graph (after text_chunks are saved)
                            # 阶段 2：实体关系提取
                            entity_relation_task = asyncio.create_task(
                                self._process_extract_entities(
                                    chunks, pipeline_status, pipeline_status_lock
                                )
                            )
                            chunk_results = await entity_relation_task
                            file_extraction_stage_ok = True

                        except Exception as e:
                            # Check if this is a user cancellation
                            if isinstance(e, PipelineCancelledException):
                                # User cancellation - log brief message only, no traceback
                                error_msg = f"User cancelled {current_file_number}/{total_files}: {file_path}"
                                logger.warning(error_msg)
                                async with pipeline_status_lock:
                                    pipeline_status["latest_message"] = error_msg
                                    pipeline_status["history_messages"].append(
                                        error_msg
                                    )
                            else:
                                # Other exceptions - log with traceback
                                logger.error(traceback.format_exc())
                                error_msg = f"Failed to extract document {current_file_number}/{total_files}: {file_path}"
                                logger.error(error_msg)
                                async with pipeline_status_lock:
                                    pipeline_status["latest_message"] = error_msg
                                    pipeline_status["history_messages"].append(
                                        traceback.format_exc()
                                    )
                                    pipeline_status["history_messages"].append(
                                        error_msg
                                    )

                            # Cancel tasks that are not yet completed
                            all_tasks = first_stage_tasks + (
                                [entity_relation_task] if entity_relation_task else []
                            )
                            for task in all_tasks:
                                if task and not task.done():
                                    task.cancel()

                            # Persistent llm cache with error handling
                            # 带错误处理的持久化 LLM 缓存
                            if self.llm_response_cache:
                                try:
                                    await self.llm_response_cache.index_done_callback()
                                except Exception as persist_error:
                                    logger.error(
                                        f"Failed to persist LLM cache: {persist_error}"
                                    )

                            # Record processing end time for failed case
                            processing_end_time = int(time.time())

                            # Update document status to failed
                            await self.doc_status.upsert(
                                {
                                    doc_id: {
                                        "status": DocStatus.FAILED,
                                        "error_msg": str(e),
                                        "content_summary": status_doc.content_summary,
                                        "content_length": status_doc.content_length,
                                        "created_at": status_doc.created_at,
                                        "updated_at": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                        "file_path": file_path,
                                        "track_id": status_doc.track_id,  # Preserve existing track_id
                                        "metadata": {
                                            "processing_start_time": processing_start_time,
                                            "processing_end_time": processing_end_time,
                                        },
                                    }
                                }
                            )

                        # Concurrency is controlled by keyed lock for individual entities and relationships
                        # 单个实体关系的并发由键控锁控制
                        if file_extraction_stage_ok:
                            try:
                                # Check for cancellation before merge
                                async with pipeline_status_lock:
                                    if pipeline_status.get(
                                        "cancellation_requested", False
                                    ):
                                        raise PipelineCancelledException(
                                            "User cancelled"
                                        )

                                # Use chunk_results from entity_relation_task
                                await merge_nodes_and_edges(
                                    chunk_results=chunk_results,  # result collected from entity_relation_task
                                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                                    entity_vdb=self.entities_vdb,
                                    relationships_vdb=self.relationships_vdb,
                                    global_config=asdict(self),
                                    full_entities_storage=self.full_entities,
                                    full_relations_storage=self.full_relations,
                                    doc_id=doc_id,
                                    pipeline_status=pipeline_status,
                                    pipeline_status_lock=pipeline_status_lock,
                                    llm_response_cache=self.llm_response_cache,
                                    entity_chunks_storage=self.entity_chunks,
                                    relation_chunks_storage=self.relation_chunks,
                                    current_file_number=current_file_number,
                                    total_files=total_files,
                                    file_path=file_path,
                                )

                                # Record processing end time
                                processing_end_time = int(time.time())

                                # 更新文档状态为已完成
                                await self.doc_status.upsert(
                                    {
                                        doc_id: {
                                            "status": DocStatus.PROCESSED,
                                            "chunks_count": len(chunks),
                                            "chunks_list": list(chunks.keys()),
                                            "content_summary": status_doc.content_summary,
                                            "content_length": status_doc.content_length,
                                            "created_at": status_doc.created_at,
                                            "updated_at": datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                            "file_path": file_path,
                                            "track_id": status_doc.track_id,  # Preserve existing track_id
                                            "metadata": {
                                                "processing_start_time": processing_start_time,
                                                "processing_end_time": processing_end_time,
                                            },
                                        }
                                    }
                                )

                                # Call _insert_done after processing each file
                                # 持久化所有存储
                                await self._insert_done()

                                async with pipeline_status_lock:
                                    log_message = f"Completed processing file {current_file_number}/{total_files}: {file_path}"
                                    logger.info(log_message)
                                    pipeline_status["latest_message"] = log_message
                                    pipeline_status["history_messages"].append(
                                        log_message
                                    )

                            except Exception as e:
                                # Check if this is a user cancellation
                                if isinstance(e, PipelineCancelledException):
                                    # User cancellation - log brief message only, no traceback
                                    error_msg = f"User cancelled during merge {current_file_number}/{total_files}: {file_path}"
                                    logger.warning(error_msg)
                                    async with pipeline_status_lock:
                                        pipeline_status["latest_message"] = error_msg
                                        pipeline_status["history_messages"].append(
                                            error_msg
                                        )
                                else:
                                    # Other exceptions - log with traceback
                                    logger.error(traceback.format_exc())
                                    error_msg = f"Merging stage failed in document {current_file_number}/{total_files}: {file_path}"
                                    logger.error(error_msg)
                                    async with pipeline_status_lock:
                                        pipeline_status["latest_message"] = error_msg
                                        pipeline_status["history_messages"].append(
                                            traceback.format_exc()
                                        )
                                        pipeline_status["history_messages"].append(
                                            error_msg
                                        )

                                # Persistent llm cache with error handling
                                if self.llm_response_cache:
                                    try:
                                        await self.llm_response_cache.index_done_callback()
                                    except Exception as persist_error:
                                        logger.error(
                                            f"Failed to persist LLM cache: {persist_error}"
                                        )

                                # Record processing end time for failed case
                                processing_end_time = int(time.time())

                                # Update document status to failed
                                await self.doc_status.upsert(
                                    {
                                        doc_id: {
                                            "status": DocStatus.FAILED,
                                            "error_msg": str(e),
                                            "content_summary": status_doc.content_summary,
                                            "content_length": status_doc.content_length,
                                            "created_at": status_doc.created_at,
                                            "updated_at": datetime.now().isoformat(),
                                            "file_path": file_path,
                                            "track_id": status_doc.track_id,  # Preserve existing track_id
                                            "metadata": {
                                                "processing_start_time": processing_start_time,
                                                "processing_end_time": processing_end_time,
                                            },
                                        }
                                    }
                                )

                # Create processing tasks for all documents
                # 对于之前所有需要处理的文档，创建处理任务，使用process_document函数
                doc_tasks = []
                for doc_id, status_doc in to_process_docs.items():
                    doc_tasks.append(
                        process_document(
                            doc_id,
                            status_doc,
                            split_by_character,
                            split_by_character_only,
                            pipeline_status,
                            pipeline_status_lock,
                            semaphore,
                        )
                    )

                # Wait for all document processing to complete
                try:
                    # 进入事件循环，开始执行任务
                    await asyncio.gather(*doc_tasks)
                except PipelineCancelledException:
                    # Cancel all remaining tasks
                    for task in doc_tasks:
                        if not task.done():
                            task.cancel()

                    # Wait for all tasks to complete cancellation
                    await asyncio.wait(doc_tasks, return_when=asyncio.ALL_COMPLETED)

                    # Exit directly (document statuses already updated in process_document)
                    return

                # Check if there's a pending request to process more documents (with lock)
                has_pending_request = False
                async with pipeline_status_lock:
                    has_pending_request = pipeline_status.get("request_pending", False)
                    if has_pending_request:
                        # Clear the request flag before checking for more documents
                        pipeline_status["request_pending"] = False

                if not has_pending_request:
                    break
                
                # 下面的逻辑都是去处理当前处理完后，但是有新的待处理请求进来的情况
                log_message = "Processing additional documents due to pending request"
                logger.info(log_message)
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

                # Check for pending documents again
                processing_docs, failed_docs, pending_docs = await asyncio.gather(
                    self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
                    self.doc_status.get_docs_by_status(DocStatus.FAILED),
                    self.doc_status.get_docs_by_status(DocStatus.PENDING),
                )
                # 更新完后，进入下一个循环，直到遇到break或者报错才退出
                to_process_docs = {}
                to_process_docs.update(processing_docs)
                to_process_docs.update(failed_docs)
                to_process_docs.update(pending_docs)

        finally:
            log_message = "Enqueued document processing pipeline stopped"
            logger.info(log_message)
            # Always reset busy status and cancellation flag when done or if an exception occurs (with lock)
            async with pipeline_status_lock:
                pipeline_status["busy"] = False
                pipeline_status["cancellation_requested"] = (
                    False  # Always reset cancellation flag
                )
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

    async def _process_extract_entities(
        self, chunk: dict[str, Any], pipeline_status=None, pipeline_status_lock=None
    ) -> list:
        """
        提取实体
        """
        try:
            chunk_results = await extract_entities(
                chunk,
                # asdict 是 Python 标准库 dataclasses 模块提供的函数，它的作用是将 dataclass 实例转换为字典。
                # asdict(self) 会将 LightRAG 类的所有字段转换为字典
                global_config=asdict(self),
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.llm_response_cache,
                text_chunks_storage=self.text_chunks,
            )
            return chunk_results
        except Exception as e:
            error_msg = f"Failed to extract entities and relationships: {str(e)}"
            logger.error(error_msg)
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = error_msg
                pipeline_status["history_messages"].append(error_msg)
            raise e

    async def _insert_done(
        self, pipeline_status=None, pipeline_status_lock=None
    ) -> None:
        """
        持久化所有存储
        """
        tasks = [
            cast(StorageNameSpace, storage_inst).index_done_callback()
            # 对于每一种存储类型，调用其持久化方法，将数据存储落盘，封装成一个task列表
            for storage_inst in [  # type: ignore
                self.full_docs,
                self.doc_status,
                self.text_chunks,
                self.full_entities,
                self.full_relations,
                self.entity_chunks,
                self.relation_chunks,
                self.llm_response_cache,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
            ]
            if storage_inst is not None
        ]
        await asyncio.gather(*tasks)

        log_message = "In memory DB persist to disk"
        logger.info(log_message)

        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

    def insert_custom_kg(
        self, custom_kg: dict[str, Any], full_doc_id: str = None
    ) -> None:
        loop = always_get_an_event_loop()
        loop.run_until_complete(self.ainsert_custom_kg(custom_kg, full_doc_id))

    async def ainsert_custom_kg(
        self,
        custom_kg: dict[str, Any],  # 自定义知识图谱数据
        full_doc_id: str = None,
    ) -> None:
        """
        插入自定义知识图谱，允许用户直接提供预处理好的实体、关系和文档块数据，绕过 LightRAG 的自动提取流程
        插入的东西有：文本块插入向量存储和文本块存储，实体和关系插入知识图谱和向量存储
        """
        update_storage = False
        try:
            # Insert chunks into vector storage
            all_chunks_data: dict[str, dict[str, str]] = {}
            chunk_to_source_map: dict[str, str] = {}    # 映射 source_id → chunk_id
            for chunk_data in custom_kg.get("chunks", []):
                chunk_content = sanitize_text_for_encoding(chunk_data["content"])
                source_id = chunk_data["source_id"]
                file_path = chunk_data.get("file_path", "custom_kg")
                tokens = len(self.tokenizer.encode(chunk_content))
                chunk_order_index = (
                    0
                    if "chunk_order_index" not in chunk_data.keys()
                    else chunk_data["chunk_order_index"]
                )
                chunk_id = compute_mdhash_id(chunk_content, prefix="chunk-")

                chunk_entry = {
                    "content": chunk_content,
                    "source_id": source_id,
                    "tokens": tokens,
                    "chunk_order_index": chunk_order_index,
                    "full_doc_id": full_doc_id
                    if full_doc_id is not None
                    else source_id,
                    "file_path": file_path,
                    "status": DocStatus.PROCESSED,
                }
                all_chunks_data[chunk_id] = chunk_entry
                # 建立 source_id 到 chunk_id 的映射
                chunk_to_source_map[source_id] = chunk_id
                update_storage = True

            if all_chunks_data:
                await asyncio.gather(
                    self.chunks_vdb.upsert(all_chunks_data),
                    self.text_chunks.upsert(all_chunks_data),
                )

            # Insert entities into knowledge graph
            all_entities_data: list[dict[str, str]] = []
            for entity_data in custom_kg.get("entities", []):
                entity_name = entity_data["entity_name"]
                entity_type = entity_data.get("entity_type", "UNKNOWN")
                description = entity_data.get("description", "No description provided")
                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")
                file_path = entity_data.get("file_path", "custom_kg")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Entity '{entity_name}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Prepare node data
                node_data: dict[str, str] = {
                    "entity_id": entity_name,
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id,
                    "file_path": file_path,
                    "created_at": int(time.time()),
                }
                # Insert node data into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_node(
                    entity_name, node_data=node_data
                )
                node_data["entity_name"] = entity_name
                all_entities_data.append(node_data)
                update_storage = True

            # Insert relationships into knowledge graph
            all_relationships_data: list[dict[str, str]] = []
            for relationship_data in custom_kg.get("relationships", []):
                src_id = relationship_data["src_id"]
                tgt_id = relationship_data["tgt_id"]
                description = relationship_data["description"]
                keywords = relationship_data["keywords"]
                weight = relationship_data.get("weight", 1.0)
                source_chunk_id = relationship_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")
                file_path = relationship_data.get("file_path", "custom_kg")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Relationship from '{src_id}' to '{tgt_id}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Check if nodes exist in the knowledge graph
                for need_insert_id in [src_id, tgt_id]:
                    if not (
                        await self.chunk_entity_relation_graph.has_node(need_insert_id)
                    ):
                        await self.chunk_entity_relation_graph.upsert_node(
                            need_insert_id,
                            node_data={
                                "entity_id": need_insert_id,
                                "source_id": source_id,
                                "description": "UNKNOWN",
                                "entity_type": "UNKNOWN",
                                "file_path": file_path,
                                "created_at": int(time.time()),
                            },
                        )

                # Insert edge into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_edge(
                    src_id,
                    tgt_id,
                    edge_data={
                        "weight": weight,
                        "description": description,
                        "keywords": keywords,
                        "source_id": source_id,
                        "file_path": file_path,
                        "created_at": int(time.time()),
                    },
                )

                edge_data: dict[str, str] = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                    "source_id": source_id,
                    "weight": weight,
                    "file_path": file_path,
                    "created_at": int(time.time()),
                }
                all_relationships_data.append(edge_data)
                update_storage = True

            # Insert entities into vector storage with consistent format
            data_for_vdb = {
                compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                    "content": dp["entity_name"] + "\n" + dp["description"],
                    "entity_name": dp["entity_name"],
                    "source_id": dp["source_id"],
                    "description": dp["description"],
                    "entity_type": dp["entity_type"],
                    "file_path": dp.get("file_path", "custom_kg"),
                }
                for dp in all_entities_data
            }
            await self.entities_vdb.upsert(data_for_vdb)

            # Insert relationships into vector storage with consistent format
            data_for_vdb = {
                compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                    "src_id": dp["src_id"],
                    "tgt_id": dp["tgt_id"],
                    "source_id": dp["source_id"],
                    "content": f"{dp['keywords']}\t{dp['src_id']}\n{dp['tgt_id']}\n{dp['description']}",
                    "keywords": dp["keywords"],
                    "description": dp["description"],
                    "weight": dp["weight"],
                    "file_path": dp.get("file_path", "custom_kg"),
                }
                for dp in all_relationships_data
            }
            await self.relationships_vdb.upsert(data_for_vdb)

        except Exception as e:
            logger.error(f"Error in ainsert_custom_kg: {e}")
            raise
        finally:
            if update_storage:
                await self._insert_done()

    def query(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | Iterator[str]:
        """
        Perform a sync query.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
            prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str: The result of the query execution.
        """
        loop = always_get_an_event_loop()

        return loop.run_until_complete(self.aquery(query, param, system_prompt))  # type: ignore

    async def aquery(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | AsyncIterator[str]:
        """异步查询接口，用于向知识图谱提问并获取 LLM 生成的回答
        Perform a async query (backward compatibility wrapper).

        This function is now a wrapper around aquery_llm that maintains backward compatibility
        by returning only the LLM response content in the original format.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
                If param.model_func is provided, it will be used instead of the global model.
            system_prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str | AsyncIterator[str]: The LLM response content.
                - Non-streaming: Returns str
                - Streaming: Returns AsyncIterator[str]
        """
        # Call the new aquery_llm function to get complete results
        result = await self.aquery_llm(query, param, system_prompt)

        # Extract and return only the LLM response for backward compatibility
        llm_response = result.get("llm_response", {})

        if llm_response.get("is_streaming"):
            return llm_response.get("response_iterator")
        else:
            return llm_response.get("content", "")

    def query_data(
        self,
        query: str,
        param: QueryParam = QueryParam(),
    ) -> dict[str, Any]:
        """调用封装aquery_data的同步接口
        Synchronous data retrieval API: returns structured retrieval results without LLM generation.

        This function is the synchronous version of aquery_data, providing the same functionality
        for users who prefer synchronous interfaces.

        Args:
            query: Query text for retrieval.
            param: Query parameters controlling retrieval behavior (same as aquery).

        Returns:
            dict[str, Any]: Same structured data result as aquery_data.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery_data(query, param))

    async def aquery_data(
        self,
        query: str,
        param: QueryParam = QueryParam(),
    ) -> dict[str, Any]:
        """
        用于获取结构化的知识图谱数据，但不调用 LLM 生成回答。
        Asynchronous data retrieval API: returns structured retrieval results without LLM generation.

        This function reuses the same logic as aquery but stops before LLM generation,
        returning the final processed entities, relationships, and chunks data that would be sent to LLM.

        Args:
            query: Query text for retrieval.
            param: Query parameters controlling retrieval behavior (same as aquery).

        Returns:
            dict[str, Any]: Structured data result in the following format:

            **Success Response:**
            ```python
            {
                "status": "success",
                "message": "Query executed successfully",
                "data": {
                    "entities": [
                        {
                            "entity_name": str,      # Entity identifier
                            "entity_type": str,      # Entity category/type
                            "description": str,      # Entity description
                            "source_id": str,        # Source chunk references
                            "file_path": str,        # Origin file path
                            "created_at": str,       # Creation timestamp
                            "reference_id": str      # Reference identifier for citations
                        }
                    ],
                    "relationships": [
                        {
                            "src_id": str,           # Source entity name
                            "tgt_id": str,           # Target entity name
                            "description": str,      # Relationship description
                            "keywords": str,         # Relationship keywords
                            "weight": float,         # Relationship strength
                            "source_id": str,        # Source chunk references
                            "file_path": str,        # Origin file path
                            "created_at": str,       # Creation timestamp
                            "reference_id": str      # Reference identifier for citations
                        }
                    ],
                    "chunks": [
                        {
                            "content": str,          # Document chunk content
                            "file_path": str,        # Origin file path
                            "chunk_id": str,         # Unique chunk identifier
                            "reference_id": str      # Reference identifier for citations
                        }
                    ],
                    "references": [
                        {
                            "reference_id": str,     # Reference identifier
                            "file_path": str         # Corresponding file path
                        }
                    ]
                },
                "metadata": {
                    "query_mode": str,           # Query mode used ("local", "global", "hybrid", "mix", "naive", "bypass")
                    "keywords": {
                        "high_level": List[str], # High-level keywords extracted
                        "low_level": List[str]   # Low-level keywords extracted
                    },
                    "processing_info": {
                        "total_entities_found": int,        # Total entities before truncation
                        "total_relations_found": int,       # Total relations before truncation
                        "entities_after_truncation": int,   # Entities after token truncation
                        "relations_after_truncation": int,  # Relations after token truncation
                        "merged_chunks_count": int,          # Chunks before final processing
                        "final_chunks_count": int            # Final chunks in result
                    }
                }
            }
            ```

            **Query Mode Differences:**
            - **local**: Focuses on entities and their related chunks based on low-level keywords
            - **global**: Focuses on relationships and their connected entities based on high-level keywords
            - **hybrid**: Combines local and global results using round-robin merging
            - **mix**: Includes knowledge graph data plus vector-retrieved document chunks
            - **naive**: Only vector-retrieved chunks, entities and relationships arrays are empty
            - **bypass**: All data arrays are empty, used for direct LLM queries

            ** processing_info is optional and may not be present in all responses, especially when query result is empty**

            **Failure Response:**
            ```python
            {
                "status": "failure",
                "message": str,  # Error description
                "data": {}       # Empty data object
            }
            ```

            **Common Failure Cases:**
            - Empty query string
            - Both high-level and low-level keywords are empty
            - Query returns empty dataset
            - Missing tokenizer or system configuration errors

        Note:
            The function adapts to the new data format from convert_to_user_format where
            actual data is nested under the 'data' field, with 'status' and 'message'
            fields at the top level.
        """
        global_config = asdict(self)

        # Create a copy of param to avoid modifying the original
        data_param = QueryParam(
            mode=param.mode,
            only_need_context=True,  # Skip LLM generation, only get context and data
            only_need_prompt=False,
            response_type=param.response_type,
            stream=False,  # Data retrieval doesn't need streaming
            top_k=param.top_k,
            chunk_top_k=param.chunk_top_k,
            max_entity_tokens=param.max_entity_tokens,
            max_relation_tokens=param.max_relation_tokens,
            max_total_tokens=param.max_total_tokens,
            hl_keywords=param.hl_keywords,
            ll_keywords=param.ll_keywords,
            conversation_history=param.conversation_history,
            history_turns=param.history_turns,
            model_func=param.model_func,
            user_prompt=param.user_prompt,
            enable_rerank=param.enable_rerank,
        )

        query_result = None

        # 这几个模式需要查询知识图谱
        if data_param.mode in ["local", "global", "hybrid", "mix"]:
            logger.debug(f"[aquery_data] Using kg_query for mode: {data_param.mode}")
            query_result = await kg_query(
                query.strip(),
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                data_param,  # Use data_param with only_need_context=True
                global_config,
                hashing_kv=self.llm_response_cache,
                system_prompt=None,
                chunks_vdb=self.chunks_vdb,
            )
        # 纯向量查询
        elif data_param.mode == "naive":
            logger.debug(f"[aquery_data] Using naive_query for mode: {data_param.mode}")
            query_result = await naive_query(
                query.strip(),
                self.chunks_vdb,
                data_param,  # Use data_param with only_need_context=True
                global_config,
                hashing_kv=self.llm_response_cache,
                system_prompt=None,
            )
        # 跳过所有检索步骤，直接将用户查询发送给LLM
        elif data_param.mode == "bypass":
            logger.debug("[aquery_data] Using bypass mode")
            # bypass mode returns empty data using convert_to_user_format
            empty_raw_data = convert_to_user_format(
                [],  # no entities
                [],  # no relationships
                [],  # no chunks
                [],  # no references
                "bypass",
            )
            query_result = QueryResult(content="", raw_data=empty_raw_data)
        else:
            raise ValueError(f"Unknown mode {data_param.mode}")

        if query_result is None:
            no_result_message = "Query returned no results"
            if data_param.mode == "naive":
                no_result_message = "No relevant document chunks found."
            final_data: dict[str, Any] = {
                "status": "failure",
                "message": no_result_message,
                "data": {},
                "metadata": {
                    "failure_reason": "no_results",
                    "mode": data_param.mode,
                },
            }
            logger.info("[aquery_data] Query returned no results.")
        else:
            # Extract raw_data from QueryResult
            # 从检索回答中提取结构化的元数据
            final_data = query_result.raw_data or {}

            # Log final result counts - adapt to new data format from convert_to_user_format
            # 记录检索到的数据量
            if final_data and "data" in final_data:
                data_section = final_data["data"]
                entities_count = len(data_section.get("entities", []))
                relationships_count = len(data_section.get("relationships", []))
                chunks_count = len(data_section.get("chunks", []))
                logger.debug(
                    f"[aquery_data] Final result: {entities_count} entities, {relationships_count} relationships, {chunks_count} chunks"
                )
            else:
                logger.warning("[aquery_data] No data section found in query result")

        await self._query_done() #持久化存储
        return final_data

    async def aquery_llm(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        完整的查询接口，返回结构化的检索结果和 LLM 生成的回答
        Asynchronous complete query API: returns structured retrieval results with LLM generation.

        This function performs a single query operation and returns both structured data and LLM response,
        based on the original aquery logic to avoid duplicate calls.

        Args:
            query: Query text for retrieval and LLM generation.
            param: Query parameters controlling retrieval and LLM behavior.
            system_prompt: Optional custom system prompt for LLM generation.

        Returns:
            dict[str, Any]: Complete response with structured data and LLM response.
        """
        logger.debug(f"[aquery_llm] Query param: {param}")

        global_config = asdict(self)

        try:
            query_result = None

            if param.mode in ["local", "global", "hybrid", "mix"]:
                query_result = await kg_query(
                    query.strip(),
                    self.chunk_entity_relation_graph,
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.text_chunks,
                    param,
                    global_config,
                    hashing_kv=self.llm_response_cache,
                    system_prompt=system_prompt,
                    chunks_vdb=self.chunks_vdb,
                )
            elif param.mode == "naive":
                query_result = await naive_query(
                    query.strip(),
                    self.chunks_vdb,
                    param,
                    global_config,
                    hashing_kv=self.llm_response_cache,
                    system_prompt=system_prompt,
                )
            elif param.mode == "bypass":
                # Bypass mode: directly use LLM without knowledge retrieval
                use_llm_func = param.model_func or global_config["llm_model_func"]
                # Apply higher priority (8) to entity/relation summary tasks
                use_llm_func = partial(use_llm_func, _priority=8)

                param.stream = True if param.stream is None else param.stream
                response = await use_llm_func(
                    query.strip(),
                    system_prompt=system_prompt,
                    history_messages=param.conversation_history,    #对话历史
                    enable_cot=True,    #启用思维链
                    stream=param.stream,
                )
                if type(response) is str:
                    return {
                        "status": "success",
                        "message": "Bypass mode LLM non streaming response",
                        "data": {},
                        "metadata": {},
                        "llm_response": {
                            "content": response,
                            "response_iterator": None,
                            "is_streaming": False,
                        },
                    }
                else:
                    return {
                        "status": "success",
                        "message": "Bypass mode LLM streaming response",
                        "data": {},
                        "metadata": {},
                        "llm_response": {
                            "content": None,
                            "response_iterator": response,
                            "is_streaming": True,
                        },
                    }
            else:
                raise ValueError(f"Unknown mode {param.mode}")

            await self._query_done()

            # Check if query_result is None
            if query_result is None:
                return {
                    "status": "failure",
                    "message": "Query returned no results",
                    "data": {},
                    "metadata": {
                        "failure_reason": "no_results",
                        "mode": param.mode,
                    },
                    "llm_response": {
                        "content": PROMPTS["fail_response"],
                        "response_iterator": None,
                        "is_streaming": False,
                    },
                }

            # Extract structured data from query result
            raw_data = query_result.raw_data or {}
            raw_data["llm_response"] = {
                "content": query_result.content
                if not query_result.is_streaming
                else None,
                "response_iterator": query_result.response_iterator
                if query_result.is_streaming
                else None,
                "is_streaming": query_result.is_streaming,
            }

            return raw_data

        except Exception as e:
            logger.error(f"Query failed: {e}")
            # Return error response
            return {
                "status": "failure",
                "message": f"Query failed: {str(e)}",
                "data": {},
                "metadata": {},
                "llm_response": {
                    "content": None,
                    "response_iterator": None,
                    "is_streaming": False,
                },
            }

    def query_llm(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        同步的查询接口，返回结构化检索数据 + LLM 生成的回答
        Synchronous complete query API: returns structured retrieval results with LLM generation.

        This function is the synchronous version of aquery_llm, providing the same functionality
        for users who prefer synchronous interfaces.

        Args:
            query: Query text for retrieval and LLM generation.
            param: Query parameters controlling retrieval and LLM behavior.
            system_prompt: Optional custom system prompt for LLM generation.

        Returns:
            dict[str, Any]: Same complete response format as aquery_llm.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery_llm(query, param, system_prompt))

    async def _query_done(self):
        await self.llm_response_cache.index_done_callback()

    async def aclear_cache(self) -> None:
        """Clear all cache data from the LLM response cache storage.

        This method clears all cached LLM responses regardless of mode.

        Example:
            # Clear all cache
            await rag.aclear_cache()
        """
        if not self.llm_response_cache:
            logger.warning("No cache storage configured")
            return

        try:
            # Clear all cache using drop method
            success = await self.llm_response_cache.drop()
            if success:
                logger.info("Cleared all cache")
            else:
                logger.warning("Failed to clear all cache")

            await self.llm_response_cache.index_done_callback()

        except Exception as e:
            logger.error(f"Error while clearing cache: {e}")

    def clear_cache(self) -> None:
        """Synchronous version of aclear_cache."""
        return always_get_an_event_loop().run_until_complete(self.aclear_cache())

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get documents by status

        Returns:
            Dict with document id is keys and document status is values
        """
        return await self.doc_status.get_docs_by_status(status)

    async def aget_docs_by_ids(
        self, ids: str | list[str]
    ) -> dict[str, DocProcessingStatus]:
        """通过文档 ID 检索一个或多个文档的处理状态,返回一个字典（ID：状态）
        Retrieves the processing status for one or more documents by their IDs.

        Args:
            ids: A single document ID (string) or a list of document IDs (list of strings).

        Returns:
            A dictionary where keys are the document IDs for which a status was found,
            and values are the corresponding DocProcessingStatus objects. IDs that
            are not found in the storage will be omitted from the result dictionary.
        """
        if isinstance(ids, str):
            # Ensure input is always a list of IDs for uniform processing
            id_list = [ids]
        elif (
            ids is None
        ):  # Handle potential None input gracefully, although type hint suggests str/list
            logger.warning(
                "aget_docs_by_ids called with None input, returning empty dict."
            )
            return {}
        else:
            # Assume input is already a list if not a string
            id_list = ids

        # Return early if the final list of IDs is empty
        if not id_list:
            logger.debug("aget_docs_by_ids called with an empty list of IDs.")
            return {}

        # Create tasks to fetch document statuses concurrently using the doc_status storage
        # 创建多个任务，list里是task，是get_by_id()的协程对象
        tasks = [self.doc_status.get_by_id(doc_id) for doc_id in id_list]
        # Execute tasks concurrently and gather the results. Results maintain order.
        # Type hint indicates results can be DocProcessingStatus or None if not found.
        results_list: list[Optional[DocProcessingStatus]] = await asyncio.gather(*tasks)

        # Build the result dictionary, mapping found IDs to their statuses
        found_statuses: dict[str, DocProcessingStatus] = {}
        # Keep track of IDs for which no status was found (for logging purposes)
        not_found_ids: list[str] = []

        # Iterate through the results, correlating them back to the original IDs
        for i, status_obj in enumerate(results_list):
            doc_id = id_list[
                i
            ]  # Get the original ID corresponding to this result index
            if status_obj:
                # If a status object was returned (not None), add it to the result dict
                found_statuses[doc_id] = status_obj
            else:
                # If status_obj is None, the document ID was not found in storage
                not_found_ids.append(doc_id)

        # Log a warning if any of the requested document IDs were not found
        if not_found_ids:
            logger.warning(
                f"Document statuses not found for the following IDs: {not_found_ids}"
            )

        # Return the dictionary containing statuses only for the found document IDs
        return found_statuses

    async def adelete_by_doc_id(
        self, doc_id: str, delete_llm_cache: bool = False
    ) -> DeletionResult:
        """文档删除核心函数,负责删除文档及其所有衍生数据,并智能重建受影响的知识图谱。
        删除文档的所有存储层数据
        分析并处理受影响的实体和关系
        使用 LLM 缓存重建部分受影响的知识图谱
        确保多进程环境下的数据一致性
        Delete a document and all its related data, including chunks, graph elements.

        This method orchestrates a comprehensive deletion process for a given document ID.
        It ensures that not only the document itself but also all its derived and associated
        data across different storage layers are removed or rebuiled. If entities or relationships
        are partially affected, they will be rebuilded using LLM cached from remaining documents.

        **Concurrency Control Design:**

        This function implements a pipeline-based concurrency control to prevent data corruption:

        1. **Single Document Deletion** (when WE acquire pipeline):
           - Sets job_name to "Single document deletion" (NOT starting with "deleting")
           - Prevents other adelete_by_doc_id calls from running concurrently
           - Ensures exclusive access to graph operations for this deletion

        2. **Batch Document Deletion** (when background_delete_documents acquires pipeline):
           - Sets job_name to "Deleting {N} Documents" (starts with "deleting")
           - Allows multiple adelete_by_doc_id calls to join the deletion queue
           - Each call validates the job name to ensure it's part of a deletion operation

        The validation logic `if not job_name.startswith("deleting") or "document" not in job_name`
        ensures that:
        - adelete_by_doc_id can only run when pipeline is idle OR during batch deletion
        - Prevents concurrent single deletions that could cause race conditions
        - Rejects operations when pipeline is busy with non-deletion tasks

        Args:
            doc_id (str): The unique identifier of the document to be deleted.
            delete_llm_cache (bool): Whether to delete cached LLM extraction results
                associated with the document. Defaults to False.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.
                - `status` (str): "success", "not_found", "not_allowed", or "failure".
                - `doc_id` (str): The ID of the document attempted to be deleted.
                - `message` (str): A summary of the operation's result.
                - `status_code` (int): HTTP status code (e.g., 200, 404, 403, 500).
                - `file_path` (str | None): The file path of the deleted document, if available.
        """
        # Get pipeline status shared data and lock for validation
        # 获取跨进程共享的管道状态
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=self.workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=self.workspace
        )

        # Track whether WE acquired the pipeline
        # 标记是否获取了管道控制权
        we_acquired_pipeline = False
        # 单文档删除: 设置 job_name="Single document deletion",阻止其他单文档删除并发
        # 批量删除: 设置 job_name="Deleting {N} Documents",允许多个删除操作并发执行
        # 非删除任务: 拒绝删除请求,返回 403

        # Check and acquire pipeline if needed
        async with pipeline_status_lock:
            if not pipeline_status.get("busy", False):
                # Pipeline is idle - WE acquire it for this deletion
                we_acquired_pipeline = True
                pipeline_status.update(
                    {
                        "busy": True,
                        "job_name": "Single document deletion",
                        "job_start": datetime.now(timezone.utc).isoformat(),
                        "docs": 1,
                        "batchs": 1,
                        "cur_batch": 0,
                        "request_pending": False,
                        "cancellation_requested": False,
                        "latest_message": f"Starting deletion for document: {doc_id}",
                    }
                )
                # Initialize history messages
                pipeline_status["history_messages"][:] = [
                    f"Starting deletion for document: {doc_id}"
                ]
            else:
                # Pipeline already busy - verify it's a deletion job
                job_name = pipeline_status.get("job_name", "").lower()
                if not job_name.startswith("deleting") or "document" not in job_name:
                    return DeletionResult(
                        status="not_allowed",
                        doc_id=doc_id,
                        message=f"Deletion not allowed: current job '{pipeline_status.get('job_name')}' is not a document deletion job",
                        status_code=403,
                        file_path=None,
                    )
                # Pipeline is busy with deletion - proceed without acquiring

        deletion_operations_started = False
        original_exception = None
        doc_llm_cache_ids: list[str] = []

        async with pipeline_status_lock:
            log_message = f"Starting deletion process for document {doc_id}"
            logger.info(log_message)
            pipeline_status["latest_message"] = log_message
            pipeline_status["history_messages"].append(log_message)

        try:
            # 1. Get the document status and related data
            doc_status_data = await self.doc_status.get_by_id(doc_id)
            file_path = doc_status_data.get("file_path") if doc_status_data else None
            if not doc_status_data:
                logger.warning(f"Document {doc_id} not found")
                return DeletionResult(
                    status="not_found",
                    doc_id=doc_id,
                    message=f"Document {doc_id} not found.",
                    status_code=404,
                    file_path="",
                )

            # Check document status and log warning for non-completed documents
            # 检查文档状态,对于未完成处理的文档记录警告日志
            raw_status = doc_status_data.get("status")
            try:
                doc_status = DocStatus(raw_status)
            except ValueError:
                doc_status = raw_status

            if doc_status != DocStatus.PROCESSED:
                if doc_status == DocStatus.PENDING:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: PENDING)"
                    )
                elif doc_status == DocStatus.PROCESSING:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: PROCESSING)"
                    )
                elif doc_status == DocStatus.PREPROCESSED:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: PREPROCESSED)"
                    )
                elif doc_status == DocStatus.FAILED:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: FAILED)"
                    )
                else:
                    status_text = (
                        doc_status.value
                        if isinstance(doc_status, DocStatus)
                        else str(doc_status)
                    )
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: {status_text})"
                    )
                logger.info(warning_msg)
                # Update pipeline status for monitoring
                # 更新管道状态以供监控
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = warning_msg
                    pipeline_status["history_messages"].append(warning_msg)

            # 2. Get chunk IDs from document status
            chunk_ids = set(doc_status_data.get("chunks_list", []))

            # 无 chunks 的文档,直接删除
            if not chunk_ids:
                logger.warning(f"No chunks found for document {doc_id}")
                # Mark that deletion operations have started
                deletion_operations_started = True
                try:
                    # Still need to delete the doc status and full doc
                    await self.full_docs.delete([doc_id])
                    await self.doc_status.delete([doc_id])
                except Exception as e:
                    logger.error(
                        f"Failed to delete document {doc_id} with no chunks: {e}"
                    )
                    raise Exception(f"Failed to delete document entry: {e}") from e

                async with pipeline_status_lock:
                    log_message = (
                        f"Document deleted without associated chunks: {doc_id}"
                    )
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)

                return DeletionResult(
                    status="success",
                    doc_id=doc_id,
                    message=log_message,
                    status_code=200,
                    file_path=file_path,
                )

            # Mark that deletion operations have started
            deletion_operations_started = True

            if delete_llm_cache and chunk_ids:
                 # 从 text_chunks 中获取每个 chunk 关联的 LLM 缓存 ID
                if not self.llm_response_cache:
                    logger.info(
                        "Skipping LLM cache collection for document %s because cache storage is unavailable",
                        doc_id,
                    )
                elif not self.text_chunks:
                    logger.info(
                        "Skipping LLM cache collection for document %s because text chunk storage is unavailable",
                        doc_id,
                    )
                else:
                    # 从 text_chunks 中获取每个 chunk 关联的 LLM 缓存 ID
                    try:
                        chunk_data_list = await self.text_chunks.get_by_ids(
                            list(chunk_ids)
                        )
                        seen_cache_ids: set[str] = set()
                        for chunk_data in chunk_data_list:
                            if not chunk_data or not isinstance(chunk_data, dict):
                                continue
                            # 获取大模型缓存 ID 列表
                            cache_ids = chunk_data.get("llm_cache_list", [])
                            if not isinstance(cache_ids, list):
                                continue
                            for cache_id in cache_ids:
                                # 确保是非空字符串且未重复添加
                                if (
                                    isinstance(cache_id, str)
                                    and cache_id
                                    and cache_id not in seen_cache_ids
                                ):
                                    doc_llm_cache_ids.append(cache_id)
                                    seen_cache_ids.add(cache_id)
                        if doc_llm_cache_ids:
                            logger.info(
                                "Collected %d LLM cache entries for document %s",
                                len(doc_llm_cache_ids),
                                doc_id,
                            )
                        else:
                            logger.info(
                                "No LLM cache entries found for document %s", doc_id
                            )
                    except Exception as cache_collect_error:
                        logger.error(
                            "Failed to collect LLM cache ids for document %s: %s",
                            doc_id,
                            cache_collect_error,
                        )
                        raise Exception(
                            f"Failed to collect LLM cache ids for document {doc_id}: {cache_collect_error}"
                        ) from cache_collect_error

            # 4. Analyze entities and relationships that will be affected
            entities_to_delete = set()  # 需要完全删除的实体
            entities_to_rebuild = {}  # entity_name -> remaining chunk id list   # 需要重建的实体 -> 剩余 chunk IDs
            relationships_to_delete = set() # 需要完全删除的关系
            relationships_to_rebuild = {}  # (src, tgt) -> remaining chunk id list   # 需要重建的关系 -> 剩余 chunk IDs
            entity_chunk_updates: dict[str, list[str]] = {}
            relation_chunk_updates: dict[tuple[str, str], list[str]] = {}

            try:
                # Get affected entities and relations from full_entities and full_relations storage
                # 从 full_entities 和 full_relations 获取受影响的元素
                doc_entities_data = await self.full_entities.get_by_id(doc_id)
                doc_relations_data = await self.full_relations.get_by_id(doc_id)

                affected_nodes = []
                affected_edges = []

                # Get entity data from graph storage using entity names from full_entities
                # 从图存储批量获取实体数据
                if doc_entities_data and "entity_names" in doc_entities_data:
                    entity_names = doc_entities_data["entity_names"]
                    # get_nodes_batch returns dict[str, dict], need to convert to list[dict]
                    # 通过实体名批量获取实体节点数据
                    nodes_dict = await self.chunk_entity_relation_graph.get_nodes_batch(
                        entity_names
                    )
                    for entity_name in entity_names:
                        # 如果实体存在于图存储中,则添加到受影响节点列表
                        node_data = nodes_dict.get(entity_name)
                        if node_data:
                            # Ensure compatibility with existing logic that expects "id" field
                            if "id" not in node_data:
                                node_data["id"] = entity_name
                            affected_nodes.append(node_data)

                # Get relation data from graph storage using relation pairs from full_relations
                # 从图存储批量获取关系数据
                if doc_relations_data and "relation_pairs" in doc_relations_data:
                    relation_pairs = doc_relations_data["relation_pairs"]
                    edge_pairs_dicts = [
                        {"src": pair[0], "tgt": pair[1]} for pair in relation_pairs
                    ]
                    # get_edges_batch returns dict[tuple[str, str], dict], need to convert to list[dict]
                    edges_dict = await self.chunk_entity_relation_graph.get_edges_batch(
                        edge_pairs_dicts
                    )

                    for pair in relation_pairs:
                        # 如果关系存在于图存储中,则添加到受影响边列表
                        src, tgt = pair[0], pair[1]
                        edge_key = (src, tgt)
                        edge_data = edges_dict.get(edge_key)
                        if edge_data:
                            # Ensure compatibility with existing logic that expects "source" and "target" fields
                            if "source" not in edge_data:
                                edge_data["source"] = src
                            if "target" not in edge_data:
                                edge_data["target"] = tgt
                            affected_edges.append(edge_data)

            except Exception as e:
                logger.error(f"Failed to analyze affected graph elements: {e}")
                raise Exception(f"Failed to analyze graph dependencies: {e}") from e

            try:
                # 处理实体依赖  affected_nodes
                # Process entities
                for node_data in affected_nodes:
                    node_label = node_data.get("entity_id")
                    if not node_label:
                        continue
                    
                    # 获取实体现有的 chunk 引用（哪些chunk里包含这个实体）
                    existing_sources: list[str] = []
                    if self.entity_chunks:
                        # 通过node_label获取存储在向量数据库中对应实体的chunk引用
                        stored_chunks = await self.entity_chunks.get_by_id(node_label)
                        if stored_chunks and isinstance(stored_chunks, dict):
                            # 从存储的chunk数据中提取chunk_ids
                            existing_sources = [
                                chunk_id
                                for chunk_id in stored_chunks.get("chunk_ids", [])
                                if chunk_id
                            ]

                    # 如果向量数据库中没有找到chunk引用,则从图数据库的属性中提取chunk引用
                    if not existing_sources and node_data.get("source_id"):
                        # Fallback: 从图数据库的属性中提取chunk引用
                        existing_sources = [
                            chunk_id
                            for chunk_id in node_data["source_id"].split(
                                GRAPH_FIELD_SEP
                            )
                            if chunk_id
                        ]

                    # 如果找了两轮，仍然没有找到chunk引用,则标记该实体为删除
                    if not existing_sources:
                        # No chunk references means this entity should be deleted
                        # 无 chunk 引用,标记为删除
                        entities_to_delete.add(node_label)
                        entity_chunk_updates[node_label] = []
                        continue
                    
                    # 计算剩余的 chunk 引用（排除将被删除的 chunk IDs）
                    remaining_sources = subtract_source_ids(existing_sources, chunk_ids)

                    if not remaining_sources:
                         # 所有 chunks 都被删除,标记为删除
                        entities_to_delete.add(node_label)
                        entity_chunk_updates[node_label] = []
                    elif remaining_sources != existing_sources:
                        # 部分 chunks 被删除,标记为重建
                        entities_to_rebuild[node_label] = remaining_sources
                        entity_chunk_updates[node_label] = remaining_sources
                    else:
                        logger.info(f"Untouch entity: {node_label}")

                async with pipeline_status_lock:
                    log_message = f"Found {len(entities_to_rebuild)} affected entities"
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)

                # 处理关系依赖  affected_edges
                # Process relationships
                for edge_data in affected_edges:
                    # source target is not in normalize order in graph db property
                    src = edge_data.get("source")
                    tgt = edge_data.get("target")

                    if not src or not tgt or "source_id" not in edge_data:
                        continue

                    edge_tuple = tuple(sorted((src, tgt)))
                    if (
                        edge_tuple in relationships_to_delete
                        or edge_tuple in relationships_to_rebuild
                    ):
                        continue

                    existing_sources: list[str] = []
                    if self.relation_chunks:
                        storage_key = make_relation_chunk_key(src, tgt)
                        stored_chunks = await self.relation_chunks.get_by_id(
                            storage_key
                        )
                        if stored_chunks and isinstance(stored_chunks, dict):
                            existing_sources = [
                                chunk_id
                                for chunk_id in stored_chunks.get("chunk_ids", [])
                                if chunk_id
                            ]

                    if not existing_sources:
                        existing_sources = [
                            chunk_id
                            for chunk_id in edge_data["source_id"].split(
                                GRAPH_FIELD_SEP
                            )
                            if chunk_id
                        ]

                    if not existing_sources:
                        # No chunk references means this relationship should be deleted
                        relationships_to_delete.add(edge_tuple)
                        relation_chunk_updates[edge_tuple] = []
                        continue

                    remaining_sources = subtract_source_ids(existing_sources, chunk_ids)

                    if not remaining_sources:
                        relationships_to_delete.add(edge_tuple)
                        relation_chunk_updates[edge_tuple] = []
                    elif remaining_sources != existing_sources:
                        relationships_to_rebuild[edge_tuple] = remaining_sources
                        relation_chunk_updates[edge_tuple] = remaining_sources
                    else:
                        logger.info(f"Untouch relation: {edge_tuple}")

                async with pipeline_status_lock:
                    log_message = (
                        f"Found {len(relationships_to_rebuild)} affected relations"
                    )
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)
                
                # 更新chunk引用存储
                current_time = int(time.time())

                # 更新 entity_chunks 存储
                if entity_chunk_updates and self.entity_chunks:
                    entity_upsert_payload = {}
                    for entity_name, remaining in entity_chunk_updates.items():
                        if not remaining:
                            # Empty entities are deleted alongside graph nodes later
                            continue
                        entity_upsert_payload[entity_name] = {
                            # 因为remaining是刚刚减法计算出来的，所以上一步就已经排除了将被删除的chunk IDs
                            "chunk_ids": remaining,
                            "count": len(remaining),
                            "updated_at": current_time,
                        }
                    if entity_upsert_payload:
                        await self.entity_chunks.upsert(entity_upsert_payload)

                # 更新 relation_chunks 存储
                if relation_chunk_updates and self.relation_chunks:
                    relation_upsert_payload = {}
                    for edge_tuple, remaining in relation_chunk_updates.items():
                        if not remaining:
                            # Empty relations are deleted alongside graph edges later
                            continue
                        storage_key = make_relation_chunk_key(*edge_tuple)
                        relation_upsert_payload[storage_key] = {
                            "chunk_ids": remaining,
                            "count": len(remaining),
                            "updated_at": current_time,
                        }

                    if relation_upsert_payload:
                        await self.relation_chunks.upsert(relation_upsert_payload)

            except Exception as e:
                logger.error(f"Failed to process graph analysis results: {e}")
                raise Exception(f"Failed to process graph dependencies: {e}") from e

            # Data integrity is ensured by allowing only one process to hold pipeline at a time（no graph db lock is needed anymore)

            # 5. Delete chunks from storage
            # 从向量数据库与文本数据库中删除chunks
            if chunk_ids:
                try:
                    await self.chunks_vdb.delete(chunk_ids)
                    await self.text_chunks.delete(chunk_ids)

                    async with pipeline_status_lock:
                        log_message = (
                            f"Successfully deleted {len(chunk_ids)} chunks from storage"
                        )
                        logger.info(log_message)
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

                except Exception as e:
                    logger.error(f"Failed to delete chunks: {e}")
                    raise Exception(f"Failed to delete document chunks: {e}") from e

            # 6. Delete relationships that have no remaining sources
            # 删除完全失效的关系
            if relationships_to_delete:
                try:
                    # Delete from relation vdb
                    rel_ids_to_delete = []
                    for src, tgt in relationships_to_delete:
                        rel_ids_to_delete.extend(
                            [
                                compute_mdhash_id(src + tgt, prefix="rel-"),
                                compute_mdhash_id(tgt + src, prefix="rel-"),
                            ]
                        )
                    await self.relationships_vdb.delete(rel_ids_to_delete)

                    # Delete from graph
                    # 删除图中的边
                    await self.chunk_entity_relation_graph.remove_edges(
                        list(relationships_to_delete)
                    )

                    # Delete from relation_chunks storage
                    # 删除 relation_chunks 存储
                    if self.relation_chunks:
                        relation_storage_keys = [
                            make_relation_chunk_key(src, tgt)
                            for src, tgt in relationships_to_delete
                        ]
                        await self.relation_chunks.delete(relation_storage_keys)

                    async with pipeline_status_lock:
                        log_message = f"Successfully deleted {len(relationships_to_delete)} relations"
                        logger.info(log_message)
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

                except Exception as e:
                    logger.error(f"Failed to delete relationships: {e}")
                    raise Exception(f"Failed to delete relationships: {e}") from e

            # 7. Delete entities that have no remaining sources
            # 删除完全失效的实体
            if entities_to_delete:
                try:
                    # Batch get all edges for entities to avoid N+1 query problem
                    # # 批量获取实体的所有边（避免 N+1 查询）
                    nodes_edges_dict = (
                        await self.chunk_entity_relation_graph.get_nodes_edges_batch(
                            list(entities_to_delete)
                        )
                    )

                    # Debug: Check and log all edges before deleting nodes
                    # 清理残留的边
                    edges_to_delete = set()
                    edges_still_exist = 0

                    # 遍历实体及其边字典以构建待删除边集合
                    for entity, edges in nodes_edges_dict.items():
                        if edges:
                            for src, tgt in edges:
                                # Normalize edge representation (sorted for consistency)

                                # 构建边的元组
                                edge_tuple = tuple(sorted((src, tgt)))
                                # 添加到待删除边集合
                                edges_to_delete.add(edge_tuple)

                                if (
                                    src in entities_to_delete
                                    and tgt in entities_to_delete
                                ):
                                    logger.warning(
                                        f"Edge still exists: {src} <-> {tgt}"
                                    )
                                elif src in entities_to_delete:
                                    logger.warning(
                                        f"Edge still exists: {src} --> {tgt}"
                                    )
                                else:
                                    logger.warning(
                                        f"Edge still exists: {src} <-- {tgt}"
                                    )
                            edges_still_exist += 1

                    if edges_still_exist:
                        logger.warning(
                            f"⚠️ {edges_still_exist} entities still has edges before deletion"
                        )

                    # Clean residual edges from VDB and storage before deleting nodes
                    if edges_to_delete:
                        # 从向量数据库删除残留边
                        # Delete from relationships_vdb
                        rel_ids_to_delete = []
                        for src, tgt in edges_to_delete:
                            rel_ids_to_delete.extend(
                                [
                                    compute_mdhash_id(src + tgt, prefix="rel-"),
                                    compute_mdhash_id(tgt + src, prefix="rel-"),
                                ]
                            )
                        await self.relationships_vdb.delete(rel_ids_to_delete)

                        # Delete from relation_chunks storage
                        # 删除 relation_chunks 存储
                        if self.relation_chunks:
                            relation_storage_keys = [
                                make_relation_chunk_key(src, tgt)
                                for src, tgt in edges_to_delete
                            ]
                            await self.relation_chunks.delete(relation_storage_keys)

                        logger.info(
                            f"Cleaned {len(edges_to_delete)} residual edges from VDB and chunk-tracking storage"
                        )

                    # Delete from graph (edges will be auto-deleted with nodes)
                    # 删除图节点（边会自动删除）
                    await self.chunk_entity_relation_graph.remove_nodes(
                        list(entities_to_delete)
                    )

                    # Delete from vector vdb
                    # 删除实体向量
                    entity_vdb_ids = [
                        compute_mdhash_id(entity, prefix="ent-")
                        for entity in entities_to_delete
                    ]
                    await self.entities_vdb.delete(entity_vdb_ids)

                    # Delete from entity_chunks storage
                    # 删除 chunk 追踪记录
                    if self.entity_chunks:
                        await self.entity_chunks.delete(list(entities_to_delete))

                    async with pipeline_status_lock:
                        log_message = (
                            f"Successfully deleted {len(entities_to_delete)} entities"
                        )
                        logger.info(log_message)
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

                except Exception as e:
                    logger.error(f"Failed to delete entities: {e}")
                    raise Exception(f"Failed to delete entities: {e}") from e

            # Persist changes to graph database before entity and relationship rebuild
            # 持久化图数据库的更改，以便后续的实体和关系重建,确保所有删除操作已写入磁盘
            await self._insert_done()

            # 8. Rebuild entities and relationships from remaining chunks
            # 重建受影响的知识图谱
            if entities_to_rebuild or relationships_to_rebuild:
                try:
                    # 重建原理:
                    # 从剩余的 chunks 中提取文本
                    # 尝试从 LLM 缓存中获取实体/关系描述
                    # 如果缓存未命中,调用 LLM 重新提取
                    # 更新图数据库和向量数据库
                    await rebuild_knowledge_from_chunks(
                        entities_to_rebuild=entities_to_rebuild,
                        relationships_to_rebuild=relationships_to_rebuild,
                        knowledge_graph_inst=self.chunk_entity_relation_graph,
                        entities_vdb=self.entities_vdb,
                        relationships_vdb=self.relationships_vdb,
                        text_chunks_storage=self.text_chunks,
                        llm_response_cache=self.llm_response_cache,
                        global_config=asdict(self),
                        pipeline_status=pipeline_status,
                        pipeline_status_lock=pipeline_status_lock,
                        entity_chunks_storage=self.entity_chunks,
                        relation_chunks_storage=self.relation_chunks,
                    )

                except Exception as e:
                    logger.error(f"Failed to rebuild knowledge from chunks: {e}")
                    raise Exception(f"Failed to rebuild knowledge graph: {e}") from e

            # 9. Delete from full_entities and full_relations storage
            try:
                await self.full_entities.delete([doc_id])
                await self.full_relations.delete([doc_id])
            except Exception as e:
                logger.error(f"Failed to delete from full_entities/full_relations: {e}")
                raise Exception(
                    f"Failed to delete from full_entities/full_relations: {e}"
                ) from e

            # 10. Delete original document and status
            try:
                await self.full_docs.delete([doc_id])
                await self.doc_status.delete([doc_id])
            except Exception as e:
                logger.error(f"Failed to delete document and status: {e}")
                raise Exception(f"Failed to delete document and status: {e}") from e

            # 删除 LLM 缓存（可选）
            if delete_llm_cache and doc_llm_cache_ids and self.llm_response_cache:
                try:
                    await self.llm_response_cache.delete(doc_llm_cache_ids)
                    cache_log_message = f"Successfully deleted {len(doc_llm_cache_ids)} LLM cache entries for document {doc_id}"
                    logger.info(cache_log_message)
                    async with pipeline_status_lock:
                        pipeline_status["latest_message"] = cache_log_message
                        pipeline_status["history_messages"].append(cache_log_message)
                    log_message = cache_log_message
                except Exception as cache_delete_error:
                    log_message = f"Failed to delete LLM cache for document {doc_id}: {cache_delete_error}"
                    logger.error(log_message)
                    logger.error(traceback.format_exc())
                    async with pipeline_status_lock:
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

            return DeletionResult(
                status="success",
                doc_id=doc_id,
                message=log_message,
                status_code=200,
                file_path=file_path,
            )

        except Exception as e:
            original_exception = e
            error_message = f"Error while deleting document {doc_id}: {e}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            return DeletionResult(
                status="fail",
                doc_id=doc_id,
                message=error_message,
                status_code=500,
                file_path=file_path,
            )

        finally:
            # ALWAYS ensure persistence if any deletion operations were started
            if deletion_operations_started:
                try:
                    await self._insert_done()
                except Exception as persistence_error:
                    persistence_error_msg = f"Failed to persist data after deletion attempt for {doc_id}: {persistence_error}"
                    logger.error(persistence_error_msg)
                    logger.error(traceback.format_exc())

                    # If there was no original exception, this persistence error becomes the main error
                    if original_exception is None:
                        return DeletionResult(
                            status="fail",
                            doc_id=doc_id,
                            message=f"Deletion completed but failed to persist changes: {persistence_error}",
                            status_code=500,
                            file_path=file_path,
                        )
                    # If there was an original exception, log the persistence error but don't override the original error
                    # The original error result was already returned in the except block
            else:
                logger.debug(
                    f"No deletion operations were started for document {doc_id}, skipping persistence"
                )

            # Release pipeline only if WE acquired it
            if we_acquired_pipeline:
                async with pipeline_status_lock:
                    pipeline_status["busy"] = False
                    pipeline_status["cancellation_requested"] = False
                    completion_msg = (
                        f"Deletion process completed for document: {doc_id}"
                    )
                    pipeline_status["latest_message"] = completion_msg
                    pipeline_status["history_messages"].append(completion_msg)
                    logger.info(completion_msg)

    async def adelete_by_entity(self, entity_name: str) -> DeletionResult:
        """异步的删除一个实体及其所有关系。
        Asynchronously delete an entity and all its relationships.

        Args:
            entity_name: Name of the entity to delete.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.
        """
        from lightrag.utils_graph import adelete_by_entity

        return await adelete_by_entity(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            entity_name,
        )

    def delete_by_entity(self, entity_name: str) -> DeletionResult:
        """Synchronously delete an entity and all its relationships.

        Args:
            entity_name: Name of the entity to delete.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_relation(
        self, source_entity: str, target_entity: str
    ) -> DeletionResult:
        """Asynchronously delete a relation between two entities.

        Args:
            source_entity: Name of the source entity.
            target_entity: Name of the target entity.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.
        """
        from lightrag.utils_graph import adelete_by_relation

        return await adelete_by_relation(
            self.chunk_entity_relation_graph,
            self.relationships_vdb,
            source_entity,
            target_entity,
        )

    def delete_by_relation(
        self, source_entity: str, target_entity: str
    ) -> DeletionResult:
        """Synchronously delete a relation between two entities.

        Args:
            source_entity: Name of the source entity.
            target_entity: Name of the target entity.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.adelete_by_relation(source_entity, target_entity)
        )

    async def get_processing_status(self) -> dict[str, int]:
        """Get current document processing status counts

        Returns:
            Dict with counts for each status
        """
        return await self.doc_status.get_status_counts()

    async def aget_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        """Get documents by track_id

        Args:
            track_id: The tracking ID to search for

        Returns:
            Dict with document id as keys and document status as values
        """
        return await self.doc_status.get_docs_by_track_id(track_id)

    async def get_entity_info(
        self, entity_name: str, include_vector_data: bool = False
    ) -> dict[str, str | None | dict[str, str]]:
        """Get detailed information of an entity"""
        from lightrag.utils_graph import get_entity_info

        return await get_entity_info(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            entity_name,
            include_vector_data,
        )

    async def get_relation_info(
        self, src_entity: str, tgt_entity: str, include_vector_data: bool = False
    ) -> dict[str, str | None | dict[str, str]]:
        """Get detailed information of a relationship"""
        from lightrag.utils_graph import get_relation_info

        return await get_relation_info(
            self.chunk_entity_relation_graph,
            self.relationships_vdb,
            src_entity,
            tgt_entity,
            include_vector_data,
        )

    async def aedit_entity(
        self,
        entity_name: str,
        updated_data: dict[str, str],
        allow_rename: bool = True,
        allow_merge: bool = False,
    ) -> dict[str, Any]:
        """Asynchronously edit entity information.

        Updates entity information in the knowledge graph and re-embeds the entity in the vector database.
        Also synchronizes entity_chunks_storage and relation_chunks_storage to track chunk references.

        Args:
            entity_name: Name of the entity to edit
            updated_data: Dictionary containing updated attributes, e.g. {"description": "new description", "entity_type": "new type"}
            allow_rename: Whether to allow entity renaming, defaults to True
            allow_merge: Whether to merge into an existing entity when renaming to an existing name

        Returns:
            Dictionary containing updated entity information
        """
        from lightrag.utils_graph import aedit_entity

        return await aedit_entity(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            entity_name,
            updated_data,
            allow_rename,
            allow_merge,
            self.entity_chunks,
            self.relation_chunks,
        )

    def edit_entity(
        self,
        entity_name: str,
        updated_data: dict[str, str],
        allow_rename: bool = True,
        allow_merge: bool = False,
    ) -> dict[str, Any]:
        """编辑实体信息的同步方法。"""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aedit_entity(entity_name, updated_data, allow_rename, allow_merge)
        )

    async def aedit_relation(
        self, source_entity: str, target_entity: str, updated_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Asynchronously edit relation information.

        Updates relation (edge) information in the knowledge graph and re-embeds the relation in the vector database.
        Also synchronizes the relation_chunks_storage to track which chunks reference this relation.

        Args:
            source_entity: Name of the source entity
            target_entity: Name of the target entity
            updated_data: Dictionary containing updated attributes, e.g. {"description": "new description", "keywords": "new keywords"}

        Returns:
            Dictionary containing updated relation information
        """
        from lightrag.utils_graph import aedit_relation

        return await aedit_relation(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            source_entity,
            target_entity,
            updated_data,
            self.relation_chunks,
        )

    def edit_relation(
        self, source_entity: str, target_entity: str, updated_data: dict[str, Any]
    ) -> dict[str, Any]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aedit_relation(source_entity, target_entity, updated_data)
        )

    async def acreate_entity(
        self, entity_name: str, entity_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Asynchronously create a new entity.

        Creates a new entity in the knowledge graph and adds it to the vector database.

        Args:
            entity_name: Name of the new entity
            entity_data: Dictionary containing entity attributes, e.g. {"description": "description", "entity_type": "type"}

        Returns:
            Dictionary containing created entity information
        """
        from lightrag.utils_graph import acreate_entity

        return await acreate_entity(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            entity_name,
            entity_data,
        )

    def create_entity(
        self, entity_name: str, entity_data: dict[str, Any]
    ) -> dict[str, Any]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.acreate_entity(entity_name, entity_data))

    async def acreate_relation(
        self, source_entity: str, target_entity: str, relation_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Asynchronously create a new relation between entities.

        Creates a new relation (edge) in the knowledge graph and adds it to the vector database.

        Args:
            source_entity: Name of the source entity
            target_entity: Name of the target entity
            relation_data: Dictionary containing relation attributes, e.g. {"description": "description", "keywords": "keywords"}

        Returns:
            Dictionary containing created relation information
        """
        from lightrag.utils_graph import acreate_relation

        return await acreate_relation(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            source_entity,
            target_entity,
            relation_data,
        )

    def create_relation(
        self, source_entity: str, target_entity: str, relation_data: dict[str, Any]
    ) -> dict[str, Any]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.acreate_relation(source_entity, target_entity, relation_data)
        )

    async def amerge_entities(
        self,
        source_entities: list[str],
        target_entity: str,
        merge_strategy: dict[str, str] = None,
        target_entity_data: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Asynchronously merge multiple entities into one entity.

        Merges multiple source entities into a target entity, handling all relationships,
        and updating both the knowledge graph and vector database.

        Args:
            source_entities: List of source entity names to merge
            target_entity: Name of the target entity after merging
            merge_strategy: Merge strategy configuration, e.g. {"description": "concatenate", "entity_type": "keep_first"}
                Supported strategies:
                - "concatenate": Concatenate all values (for text fields)
                - "keep_first": Keep the first non-empty value
                - "keep_last": Keep the last non-empty value
                - "join_unique": Join all unique values (for fields separated by delimiter)
            target_entity_data: Dictionary of specific values to set for the target entity,
                overriding any merged values, e.g. {"description": "custom description", "entity_type": "PERSON"}

        Returns:
            Dictionary containing the merged entity information
        """
        from lightrag.utils_graph import amerge_entities

        return await amerge_entities(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            source_entities,
            target_entity,
            merge_strategy,
            target_entity_data,
            self.entity_chunks,
            self.relation_chunks,
        )

    def merge_entities(
        self,
        source_entities: list[str],
        target_entity: str,
        merge_strategy: dict[str, str] = None,
        target_entity_data: dict[str, Any] = None,
    ) -> dict[str, Any]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.amerge_entities(
                source_entities, target_entity, merge_strategy, target_entity_data
            )
        )

    async def aexport_data(
        self,
        output_path: str,
        file_format: Literal["csv", "excel", "md", "txt"] = "csv",
        include_vector_data: bool = False,
    ) -> None:
        """
        Asynchronously 导出 all entities, relations, and relationships to various formats.
        Args:
            output_path: The path to the output file (including extension).
            file_format: Output format - "csv", "excel", "md", "txt".
                - csv: Comma-separated values file
                - excel: Microsoft Excel file with multiple sheets
                - md: Markdown tables
                - txt: Plain text formatted output
                - table: Print formatted tables to console
            include_vector_data: Whether to include data from the vector database.
        """
        from lightrag.utils import aexport_data as utils_aexport_data

        await utils_aexport_data(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            output_path,
            file_format,
            include_vector_data,
        )

    def export_data(
        self,
        output_path: str,
        file_format: Literal["csv", "excel", "md", "txt"] = "csv",
        include_vector_data: bool = False,
    ) -> None:
        """
        Synchronously exports all entities, relations, and relationships to various formats.
        Args:
            output_path: The path to the output file (including extension).
            file_format: Output format - "csv", "excel", "md", "txt".
                - csv: Comma-separated values file
                - excel: Microsoft Excel file with multiple sheets
                - md: Markdown tables
                - txt: Plain text formatted output
                - table: Print formatted tables to console
            include_vector_data: Whether to include data from the vector database.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(
            self.aexport_data(output_path, file_format, include_vector_data)
        )

# 项目的核心入口是LightRAG类，这是一个用@final和@dataclass装饰的主类，负责整个系统的初始化和协调。 
# 该类管理所有的存储实例、LLM调用、嵌入函数等核心组件