from __future__ import annotations
from functools import partial
from pathlib import Path

import asyncio
import json
import json_repair
from typing import Any, AsyncIterator, overload, Literal
from collections import Counter, defaultdict

from lightrag.exceptions import (
    PipelineCancelledException,
    ChunkTokenLimitExceededError,
)
from lightrag.utils import (
    logger,
    compute_mdhash_id,
    Tokenizer,
    is_float_regex,
    sanitize_and_normalize_extracted_text,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
    use_llm_func_with_cache,
    update_chunk_cache_list,
    remove_think_tags,
    pick_by_weighted_polling,
    pick_by_vector_similarity,
    process_chunks_unified,
    safe_vdb_operation_with_exception,
    create_prefixed_exception,
    fix_tuple_delimiter_corruption,
    convert_to_user_format,
    generate_reference_list_from_chunks,
    apply_source_ids_limit,
    merge_source_ids,
    make_relation_chunk_key,
)
from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
    QueryResult,
    QueryContextResult,
)
from lightrag.prompt import PROMPTS
from lightrag.constants import (
    GRAPH_FIELD_SEP,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_RELATED_CHUNK_NUMBER,
    DEFAULT_KG_CHUNK_PICK_METHOD,
    DEFAULT_ENTITY_TYPES,
    DEFAULT_SUMMARY_LANGUAGE,
    SOURCE_IDS_LIMIT_METHOD_KEEP,
    SOURCE_IDS_LIMIT_METHOD_FIFO,
    DEFAULT_FILE_PATH_MORE_PLACEHOLDER,
    DEFAULT_MAX_FILE_PATHS,
    DEFAULT_ENTITY_NAME_MAX_LENGTH,
)
from lightrag.kg.shared_storage import get_storage_keyed_lock
import time
from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)


def _truncate_entity_identifier(
    identifier: str, limit: int, chunk_key: str, identifier_role: str
) -> str:
    """用于截断过长实体标识符的辅助函数，确保实体名称/关系端点不超过配置的长度限制
    对 LLM 提取的实体/关系名称进行长度控制
    Truncate entity identifiers that exceed the configured length limit."""

    # 未超限，直接返回原值
    if len(identifier) <= limit:
        return identifier
    # 如果超限，则截取前 limit 个字符
    display_value = identifier[:limit]
    preview = identifier[:20]  # Show first 20 characters as preview
    logger.warning(
        "%s: %s len %d > %d chars (Name: '%s...')",
        chunk_key,
        identifier_role,
        len(identifier),
        limit,
        preview,
    )
    return display_value


def chunking_by_token_size(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,  ## 自定义分隔符（如 "\n\n"）
    split_by_character_only: bool = False,  # 是否仅按字符分割
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
) -> list[dict[str, Any]]:
    """默认文档分块函数，负责将长文本按 token 数量切分成可处理的小块，支持自定义分隔符和重叠策略。"""
    tokens = tokenizer.encode(content)
    results: list[dict[str, Any]] = []
    if split_by_character:
        # 先按分隔符切分
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        # 严格字符分割
        if split_by_character_only:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                # 如果仅按字符分割，且单块超限，则报错
                if len(_tokens) > chunk_token_size:
                    logger.warning(
                        "Chunk split_by_character exceeds token limit: len=%d limit=%d",
                        len(_tokens),
                        chunk_token_size,
                    )
                    raise ChunkTokenLimitExceededError(
                        chunk_tokens=len(_tokens),
                        chunk_token_limit=chunk_token_size,
                        chunk_preview=chunk[:120],
                    )
                # 未超限，直接加入结果
                new_chunks.append((len(_tokens), chunk))
        # 否则的话进行混合分割，这也是默认模式
        else:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                # 在混合模式下，如果超限。则会再按照token滑动窗口进行切分（考虑重叠的情况下切分）
                if len(_tokens) > chunk_token_size:
                    for start in range(
                        0, len(_tokens), chunk_token_size - chunk_overlap_token_size
                    ):
                        chunk_content = tokenizer.decode(
                            _tokens[start : start + chunk_token_size]
                        )
                        new_chunks.append(
                            (min(chunk_token_size, len(_tokens) - start), chunk_content)
                        )
                else:
                    new_chunks.append((len(_tokens), chunk))
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append(
                {
                    "tokens": _len,
                    "content": chunk.strip(),
                    "chunk_order_index": index,
                }
            )
    # 当没有设置分隔符，则会进行纯粹的token滑动窗口切分，这样会有用到重叠
    else:
        for index, start in enumerate(
            range(0, len(tokens), chunk_token_size - chunk_overlap_token_size)
        ):
            chunk_content = tokenizer.decode(tokens[start : start + chunk_token_size])
            results.append(
                {
                    "tokens": min(chunk_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
    return results


async def _handle_entity_relation_summary(
    description_type: str,    # "Entity" 或 "Relation"
    entity_or_relation_name: str,
    description_list: list[str],     # 待合并的描述列表
    seperator: str,  # 连接符（如 GRAPH_FIELD_SEP）
    global_config: dict,
    llm_response_cache: BaseKVStorage | None = None,
) -> tuple[str, bool]:
    """用于合并多个实体/关系描述的核心函数，采用 Map-Reduce 策略实现智能摘要，在节省 LLM 成本和保证质量之间取得平衡。
    返回的是最终的摘要描述和是否调用了 LLM 的标志。
    Handle entity relation description summary using map-reduce approach.

    This function summarizes a list of descriptions using a map-reduce strategy:
    1. If total tokens < summary_context_size and len(description_list) < force_llm_summary_on_merge, no need to summarize
    2. If total tokens < summary_max_tokens, summarize with LLM directly
    3. Otherwise, split descriptions into chunks that fit within token limits
    4. Summarize each chunk, then recursively process the summaries
    5. Continue until we get a final summary within token limits or num of descriptions is less than force_llm_summary_on_merge

    Args:
        entity_or_relation_name: Name of the entity or relation being summarized
        description_list: List of description strings to summarize
        global_config: Global configuration containing tokenizer and limits
        llm_response_cache: Optional cache for LLM responses

    Returns:
        Tuple of (final_summarized_description_string, llm_was_used_boolean)
    """
    # Handle empty input
    if not description_list:
        return "", False

    # If only one description, return it directly (no need for LLM call)
    # 单条描述直接返回（无需调用 LLM）
    if len(description_list) == 1:
        return description_list[0], False

    # Get configuration
    tokenizer: Tokenizer = global_config["tokenizer"]
    # LLM 单次处理的最大 token 数
    summary_context_size = global_config["summary_context_size"]
    # 实体/关系描述的 token 上限
    summary_max_tokens = global_config["summary_max_tokens"]
    # 低于此数量可直接拼接（无需 LLM）
    force_llm_summary_on_merge = global_config["force_llm_summary_on_merge"]

    current_list = description_list[:]  # Copy the list to avoid modifying original
    llm_was_used = False  # Track whether LLM was used during the entire process

    # Iterative map-reduce process
    while True:
        # Calculate total tokens in current list
        total_tokens = sum(len(tokenizer.encode(desc)) for desc in current_list)

        # If total length is within limits, perform final summarization
        # 当总 token 数在限制内时，或者描述条数较少时，进行最终摘要
        if total_tokens <= summary_context_size or len(current_list) <= 2:
            # 当描述数量较少且总token数较小时，无需调用LLM，直接拼接返回
            if (
                len(current_list) < force_llm_summary_on_merge
                and total_tokens < summary_max_tokens
            ):
                # no LLM needed, just join the descriptions
                final_description = seperator.join(current_list)
                return final_description if final_description else "", llm_was_used
            # 需要调用LLM进行最终摘要
            else:
                # 当描述过长且仅剩少量描述时，记录警告日志
                if total_tokens > summary_context_size and len(current_list) <= 2:
                    logger.warning(
                        f"Summarizing {entity_or_relation_name}: Oversize descpriton found"
                    )
                # Final summarization of remaining descriptions - LLM will be used
                final_summary = await _summarize_descriptions(
                    description_type,
                    entity_or_relation_name,
                    current_list,
                    global_config,
                    llm_response_cache,
                )
                return final_summary, True  # LLM was used for final summarization

        # Need to split into chunks - Map phase
        # Ensure each chunk has minimum 2 descriptions to guarantee progress

        # 更一般的状态，需要进行map phase分组
        chunks = []
        current_chunk = []
        current_tokens = 0

        # Currently least 3 descriptions in current_list
        # 当前列表中至少有3条描述
        for i, desc in enumerate(current_list):
            desc_tokens = len(tokenizer.encode(desc))

            # If adding current description would exceed limit, finalize current chunk
            # 判断是否需要结束当前分组(当前current_chunk不为空，且加入新描述会超过一组的最大限)
            if current_tokens + desc_tokens > summary_context_size and current_chunk:
                # Ensure we have at least 2 descriptions in the chunk (when possible)
                if len(current_chunk) == 1:
                    # Force add one more description to ensure minimum 2 per chunk
                    # 虽然加入会超限，但还是强制加入，确保每组至少2条描述
                    current_chunk.append(desc)
                    chunks.append(current_chunk)
                    logger.warning(
                        f"Summarizing {entity_or_relation_name}: Oversize descpriton found"
                    )
                    # 那么这一组的工作就完成了，开始下一组的处理
                    current_chunk = []  # next group is empty
                    current_tokens = 0
                else:  # curren_chunk is ready for summary in reduce phase
                    # 最开始判断了current_chunk不为空，上一个if判断了其为1，那么这里至少有两条描述
                    # 既然加入这条描述会超限，那么就加入下一组
                    # 当前组已完成，新描述放入下一组
                    chunks.append(current_chunk)
                    current_chunk = [desc]  # leave it for next group
                    current_tokens = desc_tokens
            else:
                # 继续添加描述到当前组
                current_chunk.append(desc)
                current_tokens += desc_tokens

        # Add the last chunk if it exists
        # 添加最后一组
        if current_chunk:
            chunks.append(current_chunk)

        logger.info(
            f"   Summarizing {entity_or_relation_name}: Map {len(current_list)} descriptions into {len(chunks)} groups"
        )

        # Reduce phase: summarize each group from chunks
        # 并行摘要每组描述
        # 由上面的逻辑，chuncks 中的每一组至少有两条描述，已经满足了条数的要求
        # 这里的逻辑，是满足每一组中字数的要求
        new_summaries = []
        for chunk in chunks:
            if len(chunk) == 1:
                # Optimization: single description chunks don't need LLM summarization
                # 单条描述无需 LLM 摘要，直接加入结果
                new_summaries.append(chunk[0])
            else:
                # Multiple descriptions need LLM summarization
                # 多条描述调用 LLM 进行摘要
                summary = await _summarize_descriptions(
                    description_type,
                    entity_or_relation_name,
                    chunk,
                    global_config,
                    llm_response_cache,
                )
                new_summaries.append(summary)
                llm_was_used = True  # Mark that LLM was used in reduce phase

        # Update current list with new summaries for next iteration
        # 更新当前列表，进行下一轮迭代
        current_list = new_summaries


async def _summarize_descriptions(
    description_type: str,   # "Entity" 或 "Relation"
    description_name: str,
    description_list: list[str],
    global_config: dict,
    llm_response_cache: BaseKVStorage | None = None,
) -> str:
    """Helper function to summarize a list of descriptions using LLM.

    Args:
        entity_or_relation_name: Name of the entity or relation being summarized
        descriptions: List of description strings to summarize
        global_config: Global configuration containing LLM function and settings
        llm_response_cache: Optional cache for LLM responses

    Returns:
        Summarized description string
    """
    use_llm_func: callable = global_config["llm_model_func"]
    # Apply higher priority (8) to entity/relation summary tasks
    # 设置优先级
    # _priority=8  # 实体/关系摘要（本函数）
    # _priority=5  # 实体提取
    # _priority=3  # 查询任务
    # _priority=1  # 默认任务
    use_llm_func = partial(use_llm_func, _priority=8)

    language = global_config["addon_params"].get("language", DEFAULT_SUMMARY_LANGUAGE)

    summary_length_recommended = global_config["summary_length_recommended"]

    prompt_template = PROMPTS["summarize_entity_descriptions"]

    # Convert descriptions to JSONL format and apply token-based truncation
    tokenizer = global_config["tokenizer"]
    summary_context_size = global_config["summary_context_size"]

    # Create list of JSON objects with "Description" field
    # 转为json对象列表
    json_descriptions = [{"Description": desc} for desc in description_list]

    # Use truncate_list_by_token_size for length truncation
    # 应用 token 截断（防止超过 LLM 上下文限制）
    truncated_json_descriptions = truncate_list_by_token_size(
        json_descriptions,
        key=lambda x: json.dumps(x, ensure_ascii=False),
        max_token_size=summary_context_size,
        tokenizer=tokenizer,
    )

    # Convert to JSONL format (one JSON object per line)
    # 将截断后的内容转为json格式
    joined_descriptions = "\n".join(
        json.dumps(desc, ensure_ascii=False) for desc in truncated_json_descriptions
    )

    # Prepare context for the prompt
    # 填充提示词内容
    context_base = dict(
        description_type=description_type,
        description_name=description_name,
        description_list=joined_descriptions,
        summary_length=summary_length_recommended,
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)

    # Use LLM function with cache (higher priority for summary generation)
    # 使用大模型缓存，相同输出返回缓存的结果；没有缓存就调用大模型
    summary, _ = await use_llm_func_with_cache(
        use_prompt,
        use_llm_func,
        llm_response_cache=llm_response_cache,
        cache_type="summary",
    )

    # Check summary token length against embedding limit
    embedding_token_limit = global_config.get("embedding_token_limit")
    if embedding_token_limit is not None and summary:
        tokenizer = global_config["tokenizer"]
        summary_token_count = len(tokenizer.encode(summary))
        threshold = int(embedding_token_limit)

        if summary_token_count > threshold:
            logger.warning(
                f"Summary tokens({summary_token_count}) exceeds embedding_token_limit({embedding_token_limit}) "
                f" for {description_type}: {description_name}"
            )

    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
):
    """"解析单个实体记录的核心函数，负责从 LLM 输出中提取实体信息，并进行严格的验证和清洗。
    他会满足entity<|#|>张三<|#|>person<|#|>CEO 这种类似的格式
    所以处理这种格式的数据，把信息挨个挨个提取出来"""
    if len(record_attributes) != 4 or "entity" not in record_attributes[0]:
        if len(record_attributes) > 1 and "entity" in record_attributes[0]:
            logger.warning(
                f"{chunk_key}: LLM output format error; found {len(record_attributes)}/4 feilds on ENTITY `{record_attributes[1]}` @ `{record_attributes[2] if len(record_attributes) > 2 else 'N/A'}`"
            )
            logger.debug(record_attributes)
        return None

    try:
        entity_name = sanitize_and_normalize_extracted_text(
            record_attributes[1], remove_inner_quotes=True
        )

        # entity<|#|>实体名称<|#|>实体类型<|#|>实体描述<|COMPLETE|>
        # Validate entity name after all cleaning steps
        if not entity_name or not entity_name.strip():
            logger.info(
                f"Empty entity name found after sanitization. Original: '{record_attributes[1]}'"
            )
            return None

        # Process entity type with same cleaning pipeline
        entity_type = sanitize_and_normalize_extracted_text(
            record_attributes[2], remove_inner_quotes=True
        )

        if not entity_type.strip() or any(
            char in entity_type for char in ["'", "(", ")", "<", ">", "|", "/", "\\"]
        ):
            logger.warning(
                f"Entity extraction error: invalid entity type in: {record_attributes}"
            )
            return None

        # Remove spaces and convert to lowercase
        entity_type = entity_type.replace(" ", "").lower()

        # Process entity description with same cleaning pipeline
        entity_description = sanitize_and_normalize_extracted_text(record_attributes[3])

        if not entity_description.strip():
            logger.warning(
                f"Entity extraction error: empty description for entity '{entity_name}' of type '{entity_type}'"
            )
            return None

        return dict(
            entity_name=entity_name,
            entity_type=entity_type,
            description=entity_description,
            source_id=chunk_key,
            file_path=file_path,
            timestamp=timestamp,
        )

    except ValueError as e:
        logger.error(
            f"Entity extraction failed due to encoding issues in chunk {chunk_key}: {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Entity extraction failed with unexpected error in chunk {chunk_key}: {e}"
        )
        return None


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
):
    """"解析关系记录的核心函数，负责从 LLM 输出中提取关系信息，并进行严格的验证和清洗。
    会满足 relation<|#|>张三<|#|>李四<|#|>同事<|#|>在同一家公司工作    这样的格式
    所以处理这种格式的数据，把信息挨个挨个提取出来"""
    if (
        len(record_attributes) != 5 or "relation" not in record_attributes[0]
    ):  # treat "relationship" and "relation" interchangeable
        if len(record_attributes) > 1 and "relation" in record_attributes[0]:
            logger.warning(
                f"{chunk_key}: LLM output format error; found {len(record_attributes)}/5 fields on REALTION `{record_attributes[1]}`~`{record_attributes[2] if len(record_attributes) > 2 else 'N/A'}`"
            )
            logger.debug(record_attributes)
        return None

    try:
        source = sanitize_and_normalize_extracted_text(
            record_attributes[1], remove_inner_quotes=True
        )
        target = sanitize_and_normalize_extracted_text(
            record_attributes[2], remove_inner_quotes=True
        )

        # Validate entity names after all cleaning steps
        if not source:
            logger.info(
                f"Empty source entity found after sanitization. Original: '{record_attributes[1]}'"
            )
            return None

        if not target:
            logger.info(
                f"Empty target entity found after sanitization. Original: '{record_attributes[2]}'"
            )
            return None

        if source == target:
            logger.debug(
                f"Relationship source and target are the same in: {record_attributes}"
            )
            return None

        # Process keywords with same cleaning pipeline
        edge_keywords = sanitize_and_normalize_extracted_text(
            record_attributes[3], remove_inner_quotes=True
        )
        edge_keywords = edge_keywords.replace("，", ",")

        # Process relationship description with same cleaning pipeline
        edge_description = sanitize_and_normalize_extracted_text(record_attributes[4])

        edge_source_id = chunk_key
        weight = (
            float(record_attributes[-1].strip('"').strip("'"))
            if is_float_regex(record_attributes[-1].strip('"').strip("'"))
            else 1.0
        )

        return dict(
            src_id=source,
            tgt_id=target,
            weight=weight,
            description=edge_description,
            keywords=edge_keywords,
            source_id=edge_source_id,
            file_path=file_path,
            timestamp=timestamp,
        )

    except ValueError as e:
        logger.warning(
            f"Relationship extraction failed due to encoding issues in chunk {chunk_key}: {e}"
        )
        return None
    except Exception as e:
        logger.warning(
            f"Relationship extraction failed with unexpected error in chunk {chunk_key}: {e}"
        )
        return None


async def rebuild_knowledge_from_chunks(
    entities_to_rebuild: dict[str, list[str]],  # 实体名 -> 剩余chunk_ids列表
    relationships_to_rebuild: dict[tuple[str, str], list[str]],     # (src, tgt) -> 剩余chunk_ids列表
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_storage: BaseKVStorage,
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
    pipeline_status: dict | None = None,
    pipeline_status_lock=None,
    entity_chunks_storage: BaseKVStorage | None = None,
    relation_chunks_storage: BaseKVStorage | None = None,
) -> None:
    """在删除文档后从缓存的 LLM 提取结果重建实体和关系的描述，避免昂贵的重复调用。
    Rebuild entity and relationship descriptions from cached extraction results with parallel processing

    This method uses cached LLM extraction results instead of calling LLM again,
    following the same approach as the insert process. Now with parallel processing
    controlled by llm_model_max_async and using get_storage_keyed_lock for data consistency.

    Args:
        entities_to_rebuild: Dict mapping entity_name -> list of remaining chunk_ids
        relationships_to_rebuild: Dict mapping (src, tgt) -> list of remaining chunk_ids
        knowledge_graph_inst: Knowledge graph storage
        entities_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        text_chunks_storage: Text chunks storage
        llm_response_cache: LLM response cache
        global_config: Global configuration containing llm_model_max_async
        pipeline_status: Pipeline status dictionary
        pipeline_status_lock: Lock for pipeline status
        entity_chunks_storage: KV storage maintaining full chunk IDs per entity
        relation_chunks_storage: KV storage maintaining full chunk IDs per relation
    """
    if not entities_to_rebuild and not relationships_to_rebuild:
        return

    # Get all referenced chunk IDs
    # 收集待重建的chunk IDs
    all_referenced_chunk_ids = set()
    for chunk_ids in entities_to_rebuild.values():
        all_referenced_chunk_ids.update(chunk_ids)
    for chunk_ids in relationships_to_rebuild.values():
        all_referenced_chunk_ids.update(chunk_ids)

    status_message = f"Rebuilding knowledge from {len(all_referenced_chunk_ids)} cached chunk extractions (parallel processing)"
    logger.info(status_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)

    # Get cached extraction results for these chunks using storage
    # cached_results： chunk_id -> [list of (extraction_result, create_time) from LLM cache sorted by create_time of the first extraction_result]
    # 从缓存中提取结果，获取的是大模型关于这些chunk_ids的响应结果
    cached_results = await _get_cached_extraction_results(
        llm_response_cache,
        all_referenced_chunk_ids,
        text_chunks_storage=text_chunks_storage,
    )

    if not cached_results:
        status_message = "No cached extraction results found, cannot rebuild"
        logger.warning(status_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = status_message
                pipeline_status["history_messages"].append(status_message)
        return

    # Process cached results to get entities and relationships for each chunk
    # 解析缓存结果并组织数据，为每个 chunk 处理多个提取结果
    chunk_entities = {}  # chunk_id -> {entity_name: [entity_data]}
    chunk_relationships = {}  # chunk_id -> {(src, tgt): [relationship_data]}

    for chunk_id, results in cached_results.items():
        try:
            # Handle multiple extraction results per chunk
            chunk_entities[chunk_id] = defaultdict(list)
            chunk_relationships[chunk_id] = defaultdict(list)

            # process multiple LLM extraction results for a single chunk_id
            for result in results:
                entities, relationships = await _rebuild_from_extraction_result(
                    text_chunks_storage=text_chunks_storage,
                    chunk_id=chunk_id,
                    extraction_result=result[0],
                    timestamp=result[1],
                )

                # Merge entities and relationships from this extraction result
                # Compare description lengths and keep the better version for the same chunk_id
                # 合并时保留描述更长的版本
                for entity_name, entity_list in entities.items():
                    if entity_name not in chunk_entities[chunk_id]:
                        # New entity for this chunk_id
                        chunk_entities[chunk_id][entity_name].extend(entity_list)
                    elif len(chunk_entities[chunk_id][entity_name]) == 0:
                        # Empty list, add the new entities
                        chunk_entities[chunk_id][entity_name].extend(entity_list)
                    else:
                        # Compare description lengths and keep the better one
                        existing_desc_len = len(
                            chunk_entities[chunk_id][entity_name][0].get(
                                "description", ""
                            )
                            or ""
                        )
                        new_desc_len = len(entity_list[0].get("description", "") or "")

                        if new_desc_len > existing_desc_len:
                            # Replace with the new entity that has longer description
                            # 同一个 chunk 可能有多次提取结果（如 gleaning），选择描述更详细的版本。
                            chunk_entities[chunk_id][entity_name] = list(entity_list)
                        # Otherwise keep existing version

                # Compare description lengths and keep the better version for the same chunk_id
                for rel_key, rel_list in relationships.items():
                    if rel_key not in chunk_relationships[chunk_id]:
                        # New relationship for this chunk_id
                        chunk_relationships[chunk_id][rel_key].extend(rel_list)
                    elif len(chunk_relationships[chunk_id][rel_key]) == 0:
                        # Empty list, add the new relationships
                        chunk_relationships[chunk_id][rel_key].extend(rel_list)
                    else:
                        # Compare description lengths and keep the better one
                        existing_desc_len = len(
                            chunk_relationships[chunk_id][rel_key][0].get(
                                "description", ""
                            )
                            or ""
                        )
                        new_desc_len = len(rel_list[0].get("description", "") or "")

                        if new_desc_len > existing_desc_len:
                            # Replace with the new relationship that has longer description
                            chunk_relationships[chunk_id][rel_key] = list(rel_list)
                        # Otherwise keep existing version

        except Exception as e:
            status_message = (
                f"Failed to parse cached extraction result for chunk {chunk_id}: {e}"
            )
            logger.info(status_message)  # Per requirement, change to info
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = status_message
                    pipeline_status["history_messages"].append(status_message)
            continue

    # Get max async tasks limit from global_config for semaphore control
    # 并行重建（两阶段任务调度）
    # 配置并发控制
    graph_max_async = global_config.get("llm_model_max_async", 4) * 2
    semaphore = asyncio.Semaphore(graph_max_async)

    # Counters for tracking progress
    rebuilt_entities_count = 0
    rebuilt_relationships_count = 0
    failed_entities_count = 0
    failed_relationships_count = 0

    # 创建重建任务
    async def _locked_rebuild_entity(entity_name, chunk_ids):
        nonlocal rebuilt_entities_count, failed_entities_count
        async with semaphore:
            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            async with get_storage_keyed_lock(
                [entity_name], namespace=namespace, enable_logging=False
            ):
                try:
                    await _rebuild_single_entity(
                        knowledge_graph_inst=knowledge_graph_inst,
                        entities_vdb=entities_vdb,
                        entity_name=entity_name,
                        chunk_ids=chunk_ids,
                        chunk_entities=chunk_entities,
                        llm_response_cache=llm_response_cache,
                        global_config=global_config,
                        entity_chunks_storage=entity_chunks_storage,
                    )
                    rebuilt_entities_count += 1
                except Exception as e:
                    failed_entities_count += 1
                    status_message = f"Failed to rebuild `{entity_name}`: {e}"
                    logger.info(status_message)  # Per requirement, change to info
                    if pipeline_status is not None and pipeline_status_lock is not None:
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = status_message
                            pipeline_status["history_messages"].append(status_message)

    async def _locked_rebuild_relationship(src, tgt, chunk_ids):
        nonlocal rebuilt_relationships_count, failed_relationships_count
        async with semaphore:
            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            # Sort src and tgt to ensure order-independent lock key generation
            sorted_key_parts = sorted([src, tgt])
            async with get_storage_keyed_lock(
                sorted_key_parts,
                namespace=namespace,
                enable_logging=False,
            ):
                try:
                    await _rebuild_single_relationship(
                        knowledge_graph_inst=knowledge_graph_inst,
                        relationships_vdb=relationships_vdb,
                        entities_vdb=entities_vdb,
                        src=src,
                        tgt=tgt,
                        chunk_ids=chunk_ids,
                        chunk_relationships=chunk_relationships,
                        llm_response_cache=llm_response_cache,
                        global_config=global_config,
                        relation_chunks_storage=relation_chunks_storage,
                        entity_chunks_storage=entity_chunks_storage,
                        pipeline_status=pipeline_status,
                        pipeline_status_lock=pipeline_status_lock,
                    )
                    rebuilt_relationships_count += 1
                except Exception as e:
                    failed_relationships_count += 1
                    status_message = f"Failed to rebuild `{src}`~`{tgt}`: {e}"
                    logger.info(status_message)  # Per requirement, change to info
                    if pipeline_status is not None and pipeline_status_lock is not None:
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = status_message
                            pipeline_status["history_messages"].append(status_message)

    # Create tasks for parallel processing
    tasks = []
    # 对同一实体/关系的操作串行化（避免竞态条件）
    # Add entity rebuilding tasks
    # 添加实体重建任务
    for entity_name, chunk_ids in entities_to_rebuild.items():
        task = asyncio.create_task(_locked_rebuild_entity(entity_name, chunk_ids))
        tasks.append(task)

    # Add relationship rebuilding tasks
    # 添加关系重建任务
    for (src, tgt), chunk_ids in relationships_to_rebuild.items():
        task = asyncio.create_task(_locked_rebuild_relationship(src, tgt, chunk_ids))
        tasks.append(task)

    # Log parallel processing start
    status_message = f"Starting parallel rebuild of {len(entities_to_rebuild)} entities and {len(relationships_to_rebuild)} relationships (async: {graph_max_async})"
    logger.info(status_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)

    # Execute all tasks in parallel with semaphore control and early failure detection
    # 执行任务并处理异常
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    # Check if any task raised an exception and ensure all exceptions are retrieved
    first_exception = None

    for task in done:
        try:
            exception = task.exception()
            if exception is not None:
                if first_exception is None:
                    first_exception = exception
            else:
                # Task completed successfully, retrieve result to mark as processed
                task.result()
        except Exception as e:
            if first_exception is None:
                first_exception = e

    # If any task failed, cancel all pending tasks and raise the first exception
    # Fail-Fast 策略：任何一个任务失败立即停止所有处理
    if first_exception is not None:
        # Cancel all pending tasks
        for pending_task in pending:
            pending_task.cancel()

        # Wait for cancellation to complete
        if pending:
            await asyncio.wait(pending)

        # Re-raise the first exception to notify the caller
        raise first_exception

    # Final status report
    status_message = f"KG rebuild completed: {rebuilt_entities_count} entities and {rebuilt_relationships_count} relationships rebuilt successfully."
    if failed_entities_count > 0 or failed_relationships_count > 0:
        status_message += f" Failed: {failed_entities_count} entities, {failed_relationships_count} relationships."

    logger.info(status_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)


async def _get_cached_extraction_results(
    llm_response_cache: BaseKVStorage,
    chunk_ids: set[str],
    text_chunks_storage: BaseKVStorage,
) -> dict[str, list[str]]:
    """缓存检索函数，负责从 LLM 响应缓存中提取历史的实体/关系提取结果，避免重新调用 LLM
        返回格式：
    {
        chunk_id: [(extraction_result, create_time), ...]  # 按时间排序
    }
    整个字典按每个 chunk 的最早时间排序
    Get cached extraction results for specific chunk IDs

    This function retrieves cached LLM extraction results for the given chunk IDs and returns
    them sorted by creation time. The results are sorted at two levels:
    1. Individual extraction results within each chunk are sorted by create_time (earliest first)
    2. Chunks themselves are sorted by the create_time of their earliest extraction result

    Args:
        llm_response_cache: LLM response cache storage
        chunk_ids: Set of chunk IDs to get cached results for
        text_chunks_storage: Text chunks storage for retrieving chunk data and LLM cache references

    Returns:
        Dict mapping chunk_id -> list of extraction_result_text, where:
        - Keys (chunk_ids) are ordered by the create_time of their first extraction result
        - Values (extraction results) are ordered by create_time within each chunk
    """
    cached_results = {}

    # Collect all LLM cache IDs from chunks
    # 从文本块存储中收集 LLM 缓存 IDs
    all_cache_ids = set()

    # Read from storage
    # 通过chunk_id 从text_chunks_storage里获取chunk的数据
    chunk_data_list = await text_chunks_storage.get_by_ids(list(chunk_ids))
    # 对于每一个chunk获取其llm缓存的id，再装入all_cache_ids这个set里
    for chunk_data in chunk_data_list:
        if chunk_data and isinstance(chunk_data, dict):
            # 一个 chunk 可能有多个缓存条目（初始提取 + gleaning 提取）
            # llm_cache_list 记录了所有相关的 LLM 调用缓存 ID
            llm_cache_list = chunk_data.get("llm_cache_list", [])
            if llm_cache_list:
                all_cache_ids.update(llm_cache_list)
        else:
            logger.warning(f"Chunk data is invalid or None: {chunk_data}")

    if not all_cache_ids:
        logger.warning(f"No LLM cache IDs found for {len(chunk_ids)} chunk IDs")
        return cached_results

    # Batch get LLM cache entries
    # 通过刚刚得到的大模型缓存id列表
    # 批量读取 LLM 缓存条目，也就是真实的大模型缓存内容
    cache_data_list = await llm_response_cache.get_by_ids(list(all_cache_ids))

    # Process cache entries and group by chunk_id
    
    valid_entries = 0
    # 对于每条大模型缓存条目，过滤有效条目并按 chunk 分组
    # 操作完成后，得到一个{chunk_id ：cache_data} 的字典
    for cache_entry in cache_data_list:
        if (
            cache_entry is not None
            and isinstance(cache_entry, dict)
            and cache_entry.get("cache_type") == "extract"
            # 确保是目标chunk
            and cache_entry.get("chunk_id") in chunk_ids
        ):
            chunk_id = cache_entry["chunk_id"]
            extraction_result = cache_entry["return"]
            create_time = cache_entry.get(
                "create_time", 0
            )  # Get creation time, default to 0
            valid_entries += 1

            # Support multiple LLM caches per chunk
            if chunk_id not in cached_results:
                cached_results[chunk_id] = []
            # Store tuple with extraction result and creation time for sorting
            # 存储为元组：(提取结果, 创建时间)
            cached_results[chunk_id].append((extraction_result, create_time))

    # Sort extraction results by create_time for each chunk and collect earliest times
    # 两级排序
    chunk_earliest_times = {}
    for chunk_id in cached_results:
        # Sort by create_time (x[1]), then extract only extraction_result (x[0])
        # 对每个 chunk 内的提取结果按时间排序
        cached_results[chunk_id].sort(key=lambda x: x[1])
        # Store the earliest create_time for this chunk (first item after sorting)
        chunk_earliest_times[chunk_id] = cached_results[chunk_id][0][1]

    # Sort cached_results by the earliest create_time of each chunk
    # 按每个 chunk 的最早时间对 chunks 排序
    sorted_chunk_ids = sorted(
        chunk_earliest_times.keys(), key=lambda chunk_id: chunk_earliest_times[chunk_id]
    )

    # Rebuild cached_results in sorted order
    # 重建为有序字典
    sorted_cached_results = {}
    for chunk_id in sorted_chunk_ids:
        sorted_cached_results[chunk_id] = cached_results[chunk_id]

    logger.info(
        f"Found {valid_entries} valid cache entries, {len(sorted_cached_results)} chunks with results"
    )
    return sorted_cached_results  # each item: list(extraction_result, create_time)


async def _process_extraction_result(
    result: str,
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
    tuple_delimiter: str = "<|#|>", # 字段分隔符
    completion_delimiter: str = "<|COMPLETE|>", # 完成标记
) -> tuple[dict, dict]:
    """LLM 输出解析器，负责将 LLM 的原始提取结果转换为结构化的实体和关系数据，通过多层错误修复和严格验证，确保 LLM 输出能够可靠地转换为结构化知识图谱数据
    保证能够输出成 entity<|#|>张三<|#|>person<|#|>CEO 这种格式
    然后调用了其他函数，将处理好的格式数据，转换填充为字典。返回的是字典
    Process a single extraction result (either initial or gleaning)
    Args:
        result (str): The extraction result to process
        chunk_key (str): The chunk key for source tracking
        file_path (str): The file path for citation
        tuple_delimiter (str): Delimiter for tuple fields
        record_delimiter (str): Delimiter for records
        completion_delimiter (str): Delimiter for completion
    Returns:
        tuple: (nodes_dict, edges_dict) containing the extracted entities and relationships
    """
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)

    if completion_delimiter not in result:
        logger.warning(
            f"{chunk_key}: Complete delimiter can not be found in extraction result"
        )

    # Split LLL output result to records by "\n"
    # 按换行符分割记录
    records = split_string_by_multi_markers(
        result,
        ["\n", completion_delimiter, completion_delimiter.lower()],
    )

    # Fix LLM output format error which use tuple_delimiter to seperate record instead of "\n"
    # 修复 LLM 格式错误（核心逻辑），LLM 有时会错误地使用 <|#|> 作为记录分隔符而非换行符
    fixed_records = []
    for record in records:
        record = record.strip()
        if record is None:
            continue
        # 对于每一条数据，按 "<|#|>entity<|#|>" 分割
        entity_records = split_string_by_multi_markers(
            record, [f"{tuple_delimiter}entity{tuple_delimiter}"]
        )
        # 如果分完还有，那么他就是实体记录
        for entity_record in entity_records:
            # 补充缺失的 "entity" 前缀
            if not entity_record.startswith("entity") and not entity_record.startswith(
                "relation"
            ):
                entity_record = f"entity<|{entity_record}"
            # 按 "<|#|>relation<|#|>" 分割
            entity_relation_records = split_string_by_multi_markers(
                # treat "relationship" and "relation" interchangeable
                entity_record,
                [
                    f"{tuple_delimiter}relationship{tuple_delimiter}",
                    f"{tuple_delimiter}relation{tuple_delimiter}",
                ],
            )
            for entity_relation_record in entity_relation_records:
                # 补充缺失的 "relation" 前缀
                if not entity_relation_record.startswith(
                    "entity"
                ) and not entity_relation_record.startswith("relation"):
                    entity_relation_record = (
                        f"relation{tuple_delimiter}{entity_relation_record}"
                    )
                fixed_records = fixed_records + [entity_relation_record]

    if len(fixed_records) != len(records):
        logger.warning(
            f"{chunk_key}: LLM output format error; find LLM use {tuple_delimiter} as record seperators instead new-line"
        )

    # 再次处理数据，修复分隔符污染
    for record in fixed_records:
        record = record.strip()
        if record is None:
            continue

        # Fix various forms of tuple_delimiter corruption from the LLM output using the dedicated function
        # 提取核心分隔符：从 "<|#|>" 提取 "#"
        delimiter_core = tuple_delimiter[2:-2]  # Extract "#" from "<|#|>"
        # 修复各种污染形式<|##|> → <|#|>
        # <#> → <|#|>
        # 连续的 #### → <|#|>
        record = fix_tuple_delimiter_corruption(record, delimiter_core, tuple_delimiter)
        # 如果核心分隔符有大小写变体，再修复一次
        if delimiter_core != delimiter_core.lower():
            # change delimiter_core to lower case, and fix again
            delimiter_core = delimiter_core.lower()
            record = fix_tuple_delimiter_corruption(
                record, delimiter_core, tuple_delimiter
            )
        # 按分隔符分割字段
        record_attributes = split_string_by_multi_markers(record, [tuple_delimiter])

        # Try to parse as entity
        # 尝试解析为实体
        entity_data = await _handle_single_entity_extraction(
            record_attributes, chunk_key, timestamp, file_path
        )
        if entity_data is not None:
            # 截断超长名称
            truncated_name = _truncate_entity_identifier(
                entity_data["entity_name"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                chunk_key,
                "Entity name",
            )
            entity_data["entity_name"] = truncated_name
            maybe_nodes[truncated_name].append(entity_data)
            continue

        # Try to parse as relationship
        # 尝试解析为关系
        relationship_data = await _handle_single_relationship_extraction(
            record_attributes, chunk_key, timestamp, file_path
        )
        # 截断源节点和目标节点名称
        if relationship_data is not None:
            truncated_source = _truncate_entity_identifier(
                relationship_data["src_id"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                chunk_key,
                "Relation entity",
            )
            truncated_target = _truncate_entity_identifier(
                relationship_data["tgt_id"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                chunk_key,
                "Relation entity",
            )
            relationship_data["src_id"] = truncated_source
            relationship_data["tgt_id"] = truncated_target
            maybe_edges[(truncated_source, truncated_target)].append(relationship_data)

    return dict(maybe_nodes), dict(maybe_edges)


async def _rebuild_from_extraction_result(
    text_chunks_storage: BaseKVStorage,
    extraction_result: str,
    chunk_id: str,
    timestamp: int,
) -> tuple[dict, dict]:
    """通过chunk_id得到文件路径后，解析缓存的提取结果，使用与extract_entities相同的逻辑
    Parse cached extraction result using the same logic as extract_entities

    Args:
        text_chunks_storage: Text chunks storage to get chunk data
        extraction_result: The cached LLM extraction result
        chunk_id: The chunk ID for source tracking

    Returns:
        Tuple of (entities_dict, relationships_dict)
    """

    # Get chunk data for file_path from storage
    chunk_data = await text_chunks_storage.get_by_id(chunk_id)
    file_path = (
        chunk_data.get("file_path", "unknown_source")
        if chunk_data
        else "unknown_source"
    )

    # Call the shared processing function
    return await _process_extraction_result(
        extraction_result,
        chunk_id,
        timestamp,
        file_path,
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
    )


async def _rebuild_single_entity(
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    entity_name: str, # 要重建的实体名
    chunk_ids: list[str],   # 要重建的实体所关联的 chunk id 列表
    chunk_entities: dict,   # 结构：{chunk_id: {entity_name: [entity_data_list]}}
    llm_response_cache: BaseKVStorage,  # LLM响应缓存，用于智能摘要时复用
    global_config: dict[str, str],
    entity_chunks_storage: BaseKVStorage | None = None, # 专门存储实体完整chunk列表的KV存储
    pipeline_status: dict | None = None,
    pipeline_status_lock=None,
) -> None:
    """从缓存的 LLM 提取结果中重建单个实体的完整信息（描述、类型、来源等），并同步更新到图数据库和向量库
    Rebuild a single entity from cached extraction results"""

    # Get current entity data
    # 先从图数据库中读取当前实体的现有数据
    current_entity = await knowledge_graph_inst.get_node(entity_name)
    # 没有现存 node 就直接放弃重建（该函数不负责“凭空创建实体”，创建通常发生在插入流程或关系重建时的补点逻辑）
    # 新实体的创建通常发生在：
    # 1. 初次插入文档时的 merge_nodes_then_upsert
    # 2. 关系重建时的端点补建逻辑（_rebuild_single_relationship）

    # 重建流程是在删除文档后触发的，理论上实体应该已经存在
    # 如果不存在，说明该实体已被完全删除（所有相关文档都被删除）
    # 此时不应该重建它，因为没有数据支撑
    if not current_entity:
        return

    # Helper function to update entity in both graph and vector storage
    # 一次性把结果写回图 + 向量库
    async def _update_entity_storage(
        final_description: str,
        entity_type: str,
        file_paths: list[str],
        source_chunk_ids: list[str],
        truncation_info: str = "",
    ):
        try:
            # Update entity in graph storage (critical path)
            # 组装 updated_entity_data，更新字段
            updated_entity_data = {
                **current_entity,   # 展开当前实体的所有字段（保留未修改的字段）
                "description": final_description,   # 新的合并描述
                "entity_type": entity_type,     # 新的实体类型
                "source_id": GRAPH_FIELD_SEP.join(source_chunk_ids),
                "file_path": GRAPH_FIELD_SEP.join(file_paths)
                if file_paths
                else current_entity.get("file_path", "unknown_source"),
                "created_at": int(time.time()),
                "truncate": truncation_info,
            }
            # 更新图数据库
            await knowledge_graph_inst.upsert_node(entity_name, updated_entity_data)

            # Update entity in vector database (equally critical)
            # 更新向量库：ent- 前缀的 mdhash id，content 为 "{entity_name}\n{final_description}"

            # 计算实体在向量库中的唯一ID
            # 使用MD5哈希 + "ent-"前缀，确保唯一性
            entity_vdb_id = compute_mdhash_id(entity_name, prefix="ent-")

            # 向量库中存储的content格式：
            # "{实体名}\n{实体描述}"
            entity_content = f"{entity_name}\n{final_description}"

            # 组装数据
            vdb_data = {
                entity_vdb_id: {    # 键是唯一ID
                    "content": entity_content,  # 用于生成embedding的内容
                    "entity_name": entity_name,
                    "source_id": updated_entity_data["source_id"],
                    "description": final_description,
                    "entity_type": entity_type,
                    "file_path": updated_entity_data["file_path"],
                }
            }

            # Use safe operation wrapper - VDB failure must throw exception
            # 向量库写失败会抛异常（wrapper 明确“must throw exception”），让上层并行 rebuild 任务统计失败/触发 fail-fast
            await safe_vdb_operation_with_exception(
                operation=lambda: entities_vdb.upsert(vdb_data),
                operation_name="rebuild_entity_upsert",
                entity_name=entity_name,
                max_retries=3,
                retry_delay=0.1,
            )

            # 为什么向量库失败必须抛异常？

            # 图数据库和向量库必须保持一致性
            # 如果向量库写入失败但不抛异常，会导致：
            # 图中有实体，但向量库中搜不到
            # 查询时返回不完整结果
            # Fail-Fast策略确保问题立即暴露
        except Exception as e:
            error_msg = f"Failed to update entity storage for `{entity_name}`: {e}"
            logger.error(error_msg)
            raise  # Re-raise exception

    # normalized_chunk_ids = merge_source_ids([], chunk_ids)
    # 规范化chunk ID列表
    normalized_chunk_ids = chunk_ids
    # 记录完整的chunk列表
    # 如果配置了entity_chunks_storage（专门存储实体 即entity_name - chunk_ids映射的KV存储），需要更新这里面的数据
    if entity_chunks_storage is not None and normalized_chunk_ids:
        # 保存实体的完整chunk ID列表（未截断版本）
        # 为什么要单独保存完整列表？
        # 1. 图数据库中的source_id字段会被截断（受max_source_ids_per_entity限制）
        # 2. 完整列表用于未来的增量更新和审计追踪
        # 3. 便于统计和分析（知道实体真实关联了多少个chunk）
        await entity_chunks_storage.upsert(
            {
                entity_name: {
                    "chunk_ids": normalized_chunk_ids,
                    "count": len(normalized_chunk_ids),
                }
            }
        )

    # 获取限制策略（KEEP或FIFO）
    # 如果配置中没有指定，默认使用KEEP（保留最早的）
    limit_method = (
        global_config.get("source_ids_limit_method") or SOURCE_IDS_LIMIT_METHOD_KEEP
    )
    # 应用限制，截断chunk ID列表
    # 参数说明：
    # - normalized_chunk_ids: 完整的chunk ID列表
    # - max_source_ids_per_entity: 配置的上限（如50）
    # - limit_method: KEEP（保留头部）或FIFO（保留尾部）
    # - identifier: 用于日志输出的标识符

#     前50个chunk通常已经包含实体的核心信息
# 后续chunk往往是重复提及或补充细节
# 边际收益递减：第51-120个chunk带来的新信息越来越少

# 所以因为这里的存在，其实就是会导致它的信息相对来说不会那么完整，因为没有整合所有chunk的description
    limited_chunk_ids = apply_source_ids_limit(
        normalized_chunk_ids,
        global_config["max_source_ids_per_entity"],     #配置的上限
        limit_method,    # KEEP（保留最早）或 FIFO（保留最新）
        identifier=f"`{entity_name}`",
    )

    #     为什么要限制来源数量？
    # 存储成本：图数据库的字段大小有限
    # 查询性能：过多来源会拖慢检索速度
    # 可读性：UI展示时不应该显示成百上千个来源

    # Collect all entity data from relevant (limited) chunks

    # 从chunk_entities中收集该实体在各个chunk中的所有数据

    # chunk_entities的结构：
    # {
    #     "chunk-abc": {
    #         "张三": [
    #             {"description": "...", "entity_type": "PERSON", ...},
    #             {"description": "...", "entity_type": "PERSON", ...}
    #         ]
    #     },
    #     "chunk-def": {...}
    # }
    all_entity_data = []
    # 只遍历限制后的chunk列表
    for chunk_id in limited_chunk_ids:
        # 检查该chunk是否包含该实体的数据
        if chunk_id in chunk_entities and entity_name in chunk_entities[chunk_id]:
            # 将该chunk中的所有实体数据追加到列表
            all_entity_data.extend(chunk_entities[chunk_id][entity_name])

    # 回退策略 - 从关系推断实体
    # 如果没有收集到任何实体数据（可能原因：缓存数据丢失）
    if not all_entity_data:
        logger.warning(
            f"No entity data found for `{entity_name}`, trying to rebuild from relationships"
        )

        # Get all edges connected to this entity
        # 获取与该实体相连的所有边

        # === 回退方案：从关系推断实体描述 ===
    
        # 获取该实体作为端点的所有边（关系）
        # edges的格式：[(src, tgt), (src, tgt), ...]
        # 例如：[("张三", "李四"), ("张三", "王五")]
        edges = await knowledge_graph_inst.get_node_edges(entity_name)
        # 如果连关系也没有，直接退出，放弃重建
        if not edges:
            logger.warning(f"No relations attached to entity `{entity_name}`")
            return

        # Collect relationship data to extract entity information
        # 收集关系描述和文件路径
        relationship_descriptions = []
        # 使用set自动去重
        file_paths = set()

        # Get edge data for all connected relationships
        # 从边的描述中推断实体描述
        # 遍历所有边，提取有用信息
        for src_id, tgt_id in edges:
            # 读取边的数据
            edge_data = await knowledge_graph_inst.get_edge(src_id, tgt_id)
            if edge_data:
                # 提取关系描述
                if edge_data.get("description"):
                    relationship_descriptions.append(edge_data["description"])

                # 提取文件路径
                if edge_data.get("file_path"):
                    edge_file_paths = edge_data["file_path"].split(GRAPH_FIELD_SEP)
                    file_paths.update(edge_file_paths)

        # deduplicate descriptions
        # 对描述列表去重（保持顺序）
        description_list = list(dict.fromkeys(relationship_descriptions))

        # Generate final description from relationships or fallback to current
        # 生成最终描述
        if description_list:
            # 用关系描述生成实体描述
            # 调用智能摘要函数
            final_description, _ = await _handle_entity_relation_summary(
                "Entity",
                entity_name,
                description_list,
                GRAPH_FIELD_SEP,
                global_config,
                llm_response_cache=llm_response_cache,
            )
        else:
            # 如果连关系描述也没有，使用当前值
            final_description = current_entity.get("description", "")

        # 实体类型保持不变
        entity_type = current_entity.get("entity_type", "UNKNOWN")
        # 调用helper函数写入存储
        await _update_entity_storage(
            final_description,
            entity_type,
            file_paths,
            limited_chunk_ids,
        )
        return

    # 上两步都是为了装信息进入all_entity_data，下面对装入了的信息进行处理

    # Process cached entity data
    # 如果有实体数据，进入正常处理流程
    # 初始化三个列表用于收集数据

    descriptions = []   #实体描述列表
    entity_types = []   #实体类型列表
    file_paths_list = []    #文件路径列表
    seen_paths = set()      # 用于去重文件路径

    for entity_data in all_entity_data:
        # 提取描述字段（如果存在）
        if entity_data.get("description"):
            descriptions.append(entity_data["description"])
        # 提取类型字段（如果存在）
        if entity_data.get("entity_type"):
            entity_types.append(entity_data["entity_type"])
        # 提取文件路径（带去重逻辑）
        if entity_data.get("file_path"):
            file_path = entity_data["file_path"]
            # 只添加未见过的路径
            if file_path and file_path not in seen_paths:
                file_paths_list.append(file_path)
                seen_paths.add(file_path)

    # Apply MAX_FILE_PATHS limit
    # 应用文件路径数量限制
    max_file_paths = global_config.get("max_file_paths")
    file_path_placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )
    limit_method = global_config.get("source_ids_limit_method")

    # 记录原始数量
    original_count = len(file_paths_list)
    # 如果超过限制
    if original_count > max_file_paths:
        # 根据策略截断
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            # FIFO策略：保留尾部（最新的）
            file_paths_list = file_paths_list[-max_file_paths:]
        else:
            # KEEP: keep head (earliest), discard tail
            # KEEP策略：保留头部（最早的）
            file_paths_list = file_paths_list[:max_file_paths]

        # 追加占位符，表示还有更多
        file_paths_list.append(
            f"...{file_path_placeholder}...({limit_method} {max_file_paths}/{original_count})"
        )
        logger.info(
            f"Limited `{entity_name}`: file_path {original_count} -> {max_file_paths} ({limit_method})"
        )

    # Remove duplicates while preserving order
    # 对描述列表去重（保持顺序）
    description_list = list(dict.fromkeys(descriptions))
    # 对类型列表去重（保持顺序）
    entity_types = list(dict.fromkeys(entity_types))

    # Get most common entity type
    # 实体类型投票：选择出现次数最多的类型
    entity_type = (
        max(set(entity_types), key=entity_types.count)
        if entity_types
        else current_entity.get("entity_type", "UNKNOWN")
    )

    # Generate final description from entities or fallback to current
    # 描述合并
    if description_list:
        # 调用智能摘要函数合并描述
        # 这个函数的内部逻辑：
        # 1. 如果描述很少，直接拼接
        # 2. 如果描述较多，使用Map-Reduce策略
        # 3. 如果超过token限制，分批摘要后再合并
        final_description, _ = await _handle_entity_relation_summary(
            "Entity",
            entity_name,
            description_list,
            GRAPH_FIELD_SEP,
            global_config,
            llm_response_cache=llm_response_cache,
        )
    else:
        # 如果没有描述，使用当前实体的描述
        final_description = current_entity.get("description", "")

    # 如果应用了限制（实际使用的chunk数 < 原始chunk数）
    if len(limited_chunk_ids) < len(normalized_chunk_ids):
        # 生成截断信息字符串
        truncation_info = (
            f"{limit_method} {len(limited_chunk_ids)}/{len(normalized_chunk_ids)}"
        )
    else:
        # 没有截断，信息为空
        truncation_info = ""

    # 调用之前定义的helper函数，一次性更新图和向量库
    await _update_entity_storage(
        final_description,
        entity_type,
        file_paths_list,
        limited_chunk_ids,
        truncation_info,
    )

    # Log rebuild completion with truncation info
    status_message = f"Rebuild `{entity_name}` from {len(chunk_ids)} chunks"
    if truncation_info:
        status_message += f" ({truncation_info})"
    logger.info(status_message)
    # Update pipeline status
    # 更新管道状态（如果配置了）
# 这个状态会被前端轮询，用于显示进度条
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)


async def _rebuild_single_relationship(
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    entities_vdb: BaseVectorStorage,
    src: str,
    tgt: str,
    chunk_ids: list[str],       # 该关系关联的所有chunk ID列表
    chunk_relationships: dict,
    llm_response_cache: BaseKVStorage,      # LLM响应缓存
    global_config: dict[str, str],
    relation_chunks_storage: BaseKVStorage | None = None,   # 关系-chunk映射存储
    entity_chunks_storage: BaseKVStorage | None = None,     # 实体-chunk映射存储
    pipeline_status: dict | None = None,
    pipeline_status_lock=None,
) -> None:
    """Rebuild a single relationship from cached extraction results

    Note: This function assumes the caller has already acquired the appropriate
    keyed lock for the relationship pair to ensure thread safety.
    """

    # Get current relationship data
    # 从图数据库读取当前关系数据
    current_relationship = await knowledge_graph_inst.get_edge(src, tgt)
    # 如果关系不存在，直接返回
    # 设计原因：与实体重建相同，此函数负责"更新已有关系"而非"创建新关系"
    # 新关系创建发生在插入流程（_merge_edges_then_upsert）
    if not current_relationship:
        return

    # normalized_chunk_ids = merge_source_ids([], chunk_ids)
    # 直接使用传入的列表
    normalized_chunk_ids = chunk_ids
    # 如果配置了relation_chunks_storage，保存完整列表
    if relation_chunks_storage is not None and normalized_chunk_ids:
        storage_key = make_relation_chunk_key(src, tgt)
        # 跟新relation_chunks_storage库中storage_key对应的信息
        await relation_chunks_storage.upsert(
            {
                storage_key: {
                    "chunk_ids": normalized_chunk_ids,
                    "count": len(normalized_chunk_ids),
                }
            }
        )
    # 获取限制策略
    limit_method = (
        global_config.get("source_ids_limit_method") or SOURCE_IDS_LIMIT_METHOD_KEEP
    )
    # 应用限制，截断chunk ID列表，只选取部分chunk的信息用于重建关系
    limited_chunk_ids = apply_source_ids_limit(
        normalized_chunk_ids,
        global_config["max_source_ids_per_relation"],
        limit_method,
        identifier=f"`{src}`~`{tgt}`",
    )

    # Collect all relationship data from relevant chunks
    # 收集所有关系数据
    all_relationship_data = []
    # 遍历限制后的chunk列表
    for chunk_id in limited_chunk_ids:
        # 检查该chunk是否包含关系数据
        if chunk_id in chunk_relationships:
            # Check both (src, tgt) and (tgt, src) since relationships can be bidirectional
            # ⭐ 关键特性：双向检查
            # 因为图是无向的，关系可能以任一方向存储
            # 例如：(张三, 李四) 和 (李四, 张三) 应该被视为同一关系
            for edge_key in [(src, tgt), (tgt, src)]:
                if edge_key in chunk_relationships[chunk_id]:
                    # 收集该方向的关系数据
                    all_relationship_data.extend(
                        chunk_relationships[chunk_id][edge_key]
                    )
    # 如果没有收集到任何数据，记录警告并返回
    if not all_relationship_data:
        logger.warning(f"No relation data found for `{src}-{tgt}`")
        return

    # Merge descriptions and keywords
    descriptions = []   # 关系描述列表
    keywords = []       # 关键词列表
    weights = []        # 权重列表
    file_paths_list = []    # 文件路径列表
    seen_paths = set()      # 用于去重

    # 遍历所有收集到的关系数据
    for rel_data in all_relationship_data:
        if rel_data.get("description"):
            descriptions.append(rel_data["description"])
        if rel_data.get("keywords"):
            keywords.append(rel_data["keywords"])
        if rel_data.get("weight"):
            weights.append(rel_data["weight"])
        if rel_data.get("file_path"):
            file_path = rel_data["file_path"]
            if file_path and file_path not in seen_paths:
                file_paths_list.append(file_path)
                seen_paths.add(file_path)

    # Apply count limit
    # 获取配置参数
    max_file_paths = global_config.get("max_file_paths")
    file_path_placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )
    limit_method = global_config.get("source_ids_limit_method")

    # 记录原始数量
    original_count = len(file_paths_list)
    # 如果超过限制，进行截断
    if original_count > max_file_paths:
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]
        # 追加占位符
        file_paths_list.append(
            f"...{file_path_placeholder}...({limit_method} {max_file_paths}/{original_count})"
        )
        logger.info(
            f"Limited `{src}`~`{tgt}`: file_path {original_count} -> {max_file_paths} ({limit_method})"
        )

    # Remove duplicates while preserving order
    # 对描述和关键词去重
    description_list = list(dict.fromkeys(descriptions))
    keywords = list(dict.fromkeys(keywords))

    # 合并关键词（用逗号连接）
    # 例如：["同事", "朋友", "合作伙伴"] -> "同事, 朋友, 合作伙伴"
    combined_keywords = (
        ", ".join(set(keywords))
        if keywords
        else current_relationship.get("keywords", "")
    )

    # 计算总权重（累加）
    weight = sum(weights) if weights else current_relationship.get("weight", 1.0)

    # Generate final description from relations or fallback to current
    # 如果有描述列表
    if description_list:
        # 调用智能摘要函数（与实体相同）
        final_description, _ = await _handle_entity_relation_summary(
            "Relation",
            f"{src}-{tgt}",
            description_list,   # 要合并的描述列表
            GRAPH_FIELD_SEP,    # 分隔符
            global_config,
            llm_response_cache=llm_response_cache,
        )
    else:
        # fallback to keep current(unchanged)
        final_description = current_relationship.get("description", "")

    # 如果应用了限制
    # 生成截断信息
    if len(limited_chunk_ids) < len(normalized_chunk_ids):
        truncation_info = (
            f"{limit_method} {len(limited_chunk_ids)}/{len(normalized_chunk_ids)}"
        )
    else:
        truncation_info = ""

    # Update relationship in graph storage
    # 组装更新后的关系数据
    updated_relationship_data = {
        **current_relationship,     # 展开当前关系的所有字段
        # 覆盖需要更新的字段
        "description": final_description
        if final_description
        else current_relationship.get("description", ""),
        "keywords": combined_keywords,
        "weight": weight,
        "source_id": GRAPH_FIELD_SEP.join(limited_chunk_ids),
        "file_path": GRAPH_FIELD_SEP.join([fp for fp in file_paths_list if fp])
        if file_paths_list
        else current_relationship.get("file_path", "unknown_source"),
        "truncate": truncation_info,
    }

    # Ensure both endpoint nodes exist before writing the edge back
    # (certain storage backends require pre-existing nodes).
    # 端点节点检查与补建

    # 准备节点数据（用于可能的补建）
    node_description = (
        updated_relationship_data["description"]
        if updated_relationship_data.get("description")
        else current_relationship.get("description", "")
    )
    node_source_id = updated_relationship_data.get("source_id", "")
    node_file_path = updated_relationship_data.get("file_path", "unknown_source")

    # 某些图数据库（如Neo4j）要求创建边前必须先有节点
    for node_id in {src, tgt}:   # 检查源节点和目标节点
        # 如果节点不存在，则创建
        if not (await knowledge_graph_inst.has_node(node_id)):
            # 创建节点数据
            node_created_at = int(time.time())
            node_data = {
                "entity_id": node_id,
                "source_id": node_source_id,
                "description": node_description,
                "entity_type": "UNKNOWN",
                "file_path": node_file_path,
                "created_at": node_created_at,
                "truncate": "",
            }
            # 在图数据库中创建节点
            await knowledge_graph_inst.upsert_node(node_id, node_data=node_data)

            # Update entity_chunks_storage for the newly created entity
            # 更新 entity_chunks_storage
            if entity_chunks_storage is not None and limited_chunk_ids:
                await entity_chunks_storage.upsert(
                    {
                        node_id: {
                            "chunk_ids": limited_chunk_ids,
                            "count": len(limited_chunk_ids),
                        }
                    }
                )

            # Update entity_vdb for the newly created entity
            # 更新实体向量库
            if entities_vdb is not None:
                entity_vdb_id = compute_mdhash_id(node_id, prefix="ent-")
                entity_content = f"{node_id}\n{node_description}"
                vdb_data = {
                    entity_vdb_id: {
                        "content": entity_content,
                        "entity_name": node_id,
                        "source_id": node_source_id,
                        "entity_type": "UNKNOWN",
                        "file_path": node_file_path,
                    }
                }
                await safe_vdb_operation_with_exception(
                    operation=lambda payload=vdb_data: entities_vdb.upsert(payload),
                    operation_name="rebuild_added_entity_upsert",
                    entity_name=node_id,
                    max_retries=3,
                    retry_delay=0.1,
                )

    await knowledge_graph_inst.upsert_edge(src, tgt, updated_relationship_data)

    # Update relationship in vector database
    # Sort src and tgt to ensure consistent ordering (smaller string first)
    # 排序源和目标节点，确保一致性
    # 例如：("李四", "张三") -> ("张三", "李四")
    # 这样可以避免同一关系的两个方向被视为不同关系
    if src > tgt:
        src, tgt = tgt, src
    try:
        # 计算向量库ID（两个方向都要算）
        rel_vdb_id = compute_mdhash_id(src + tgt, prefix="rel-")
        rel_vdb_id_reverse = compute_mdhash_id(tgt + src, prefix="rel-")

        # Delete old vector records first (both directions to be safe)
        # ⭐ 关键操作：先删除旧记录（两个方向）
        # 为什么要删除？因为关系的方向可能改变
        try:
            await relationships_vdb.delete([rel_vdb_id, rel_vdb_id_reverse])
        except Exception as e:
            logger.debug(
                f"Could not delete old relationship vector records {rel_vdb_id}, {rel_vdb_id_reverse}: {e}"
            )

        # Insert new vector record
        # 插入新的向量记录
        # 关系向量库的content格式：
        # "{keywords}\t{src}\n{tgt}\n{description}"
        # 例如："同事, 朋友\t张三\n李四\n张三和李四是同事..."
        rel_content = f"{combined_keywords}\t{src}\n{tgt}\n{final_description}"
        vdb_data = {
            rel_vdb_id: {
                "src_id": src,
                "tgt_id": tgt,
                "source_id": updated_relationship_data["source_id"],
                "content": rel_content,
                "keywords": combined_keywords,
                "description": final_description,
                "weight": weight,
                "file_path": updated_relationship_data["file_path"],
            }
        }

        # Use safe operation wrapper - VDB failure must throw exception
        await safe_vdb_operation_with_exception(
            operation=lambda: relationships_vdb.upsert(vdb_data),
            operation_name="rebuild_relationship_upsert",
            entity_name=f"{src}-{tgt}",
            max_retries=3,
            retry_delay=0.2,
        )

    except Exception as e:
        error_msg = f"Failed to rebuild relationship storage for `{src}-{tgt}`: {e}"
        logger.error(error_msg)
        raise  # Re-raise exception

    # Log rebuild completion with truncation info
    status_message = f"Rebuild `{src}`~`{tgt}` from {len(chunk_ids)} chunks"
    if truncation_info:
        status_message += f" ({truncation_info})"
    # Add truncation info from apply_source_ids_limit if truncation occurred
    if len(limited_chunk_ids) < len(normalized_chunk_ids):
        truncation_info = (
            f" ({limit_method}:{len(limited_chunk_ids)}/{len(normalized_chunk_ids)})"
        )
        status_message += truncation_info

    logger.info(status_message)

    # Update pipeline status
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)


async def _merge_nodes_then_upsert(
    entity_name: str, # 实体名称
    nodes_data: list[dict],# 新提取的实体数据列表
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage | None,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
):
    """负责插入新文档时的实体合并逻辑，插入新文档时，将新提取的实体数据与已有数据合并
    Get existing nodes from knowledge graph use name,if exists, merge data, else create, then upsert."""
    already_entity_types = []
    already_source_ids = []
    already_description = []
    already_file_paths = []

    # 1. Get existing node data from knowledge graph
    # 从图数据库读取已有实体
    already_node = await knowledge_graph_inst.get_node(entity_name)
    # 如果实体已存在，提取其数据
    if already_node:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(already_node["source_id"].split(GRAPH_FIELD_SEP))
        already_file_paths.extend(already_node["file_path"].split(GRAPH_FIELD_SEP))
        already_description.extend(already_node["description"].split(GRAPH_FIELD_SEP))

    # 从新数据中提取chunk IDs
    # nodes_data的结构：
    # [
    #     {"entity_name": "张三", "entity_type": "PERSON", 
    #      "description": "...", "source_id": "chunk-new-1"},
    #     {"entity_name": "张三", "entity_type": "PERSON", 
    #      "description": "...", "source_id": "chunk-new-2"}
    # ]
    new_source_ids = [dp["source_id"] for dp in nodes_data if dp.get("source_id")]

    # 从 entity_chunks_storage 获取entity_name对应的chunk的id的完整列表（老的）
    existing_full_source_ids = []
    if entity_chunks_storage is not None:
        # 读取完整chunk列表（未截断版本）
        stored_chunks = await entity_chunks_storage.get_by_id(entity_name)
        if stored_chunks and isinstance(stored_chunks, dict):
            existing_full_source_ids = [
                chunk_id for chunk_id in stored_chunks.get("chunk_ids", []) if chunk_id
            ]
    # 如果没有存储完整列表，回退到从图数据库获取
    if not existing_full_source_ids:
        existing_full_source_ids = [
            chunk_id for chunk_id in already_source_ids if chunk_id
        ]

    # 2. Merging new source ids with existing ones
    # 合并新旧chunk IDs（去重、排序）

    # merge_source_ids 的作用：
    # 1. 去重：避免重复chunk
    # 2. 排序：保持时间顺序
    # 3. 规范化：统一格式
    full_source_ids = merge_source_ids(existing_full_source_ids, new_source_ids)

    # 更新 entity_chunks_storage，保证这个entity_name对应的完整chunk列表是最新的
    if entity_chunks_storage is not None and full_source_ids:
        await entity_chunks_storage.upsert(
            {
                entity_name: {
                    "chunk_ids": full_source_ids,
                    "count": len(full_source_ids),
                }
            }
        )

    # 3. Finalize source_id by applying source ids limit
    limit_method = global_config.get("source_ids_limit_method")
    max_source_limit = global_config.get("max_source_ids_per_entity")
    # 应用限制
    source_ids = apply_source_ids_limit(
        full_source_ids,
        max_source_limit,
        limit_method,
        identifier=f"`{entity_name}`",
    )

    # 4. Only keep nodes not filter by apply_source_ids_limit if limit_method is KEEP
    # 如果使用KEEP策略（保留最早的chunk）
    if limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP:
        # 创建允许的chunk ID集合
        allowed_source_ids = set(source_ids)
        filtered_nodes = []
        # 遍历新提取的实体数据
        for dp in nodes_data:
            source_id = dp.get("source_id")
            # Skip descriptions sourced from chunks dropped by the limitation cap
            # ⚠️ 关键逻辑：跳过不在允许列表中的chunk，则不用合并
            # 条件：
            # 1. source_id 存在
            # 2. 不在 allowed_source_ids（前50个）中
            # 3. 不在 existing_full_source_ids（已有的chunk）中
            if (
                source_id
                and source_id not in allowed_source_ids
                and source_id not in existing_full_source_ids
            ):
                continue
            # 保留符合条件的实体数据
            filtered_nodes.append(dp)
        # 更新nodes_data为过滤后的数据    
        nodes_data = filtered_nodes
    else:  # In FIFO mode, keep all nodes - truncation happens at source_ids level only
        # FIFO模式：保留所有新数据
        nodes_data = list(nodes_data)

    # 5. Check if we need to skip summary due to source_ids limit
    # 检查是否需要跳过摘要

    # 特殊情况：KEEP策略 + 已满 + 无新数据
    if (
        limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP
        and len(existing_full_source_ids) >= max_source_limit
        and not nodes_data
    ):
        if already_node:
            logger.info(
                f"Skipped `{entity_name}`: KEEP old chunks {already_source_ids}/{len(full_source_ids)}"
            )
            # 直接返回已有节点数据（不做任何修改）
            existing_node_data = dict(already_node)
            return existing_node_data
        else:
            logger.error(f"Internal Error: already_node missing for `{entity_name}`")
            raise ValueError(
                f"Internal Error: already_node missing for `{entity_name}`"
            )

    # 6.1 Finalize source_id
    # 最终的source_id字段（用于图数据库）
    source_id = GRAPH_FIELD_SEP.join(source_ids)

    # 6.2 Finalize entity type by highest count
    # 使用Counter统计频率，选择最常见的
    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]

    # 7. Deduplicate nodes by description, keeping first occurrence in the same document
    # 使用字典去重（保留第一次出现）
    unique_nodes = {}
    for dp in nodes_data:
        desc = dp.get("description")
        if not desc:
            continue
        # 如果描述未见过，记录
        if desc not in unique_nodes:
            # 键是描述，值是完整数据
            unique_nodes[desc] = dp

    # Sort description by timestamp, then by description length when timestamps are the same
    # 排序：先按时间戳，再按描述长度（降序）
    sorted_nodes = sorted(
        unique_nodes.values(),
        key=lambda x: (x.get("timestamp", 0), -len(x.get("description", ""))),
    )
    # 提取排序后的描述
    sorted_descriptions = [dp["description"] for dp in sorted_nodes]

    # Combine already_description with sorted new sorted descriptions
    # 合并新旧描述
    description_list = already_description + sorted_descriptions
    if not description_list:
        logger.error(f"Entity {entity_name} has no description")
        raise ValueError(f"Entity {entity_name} has no description")

    # Check for cancellation before LLM summary
    # 在调用LLM前检查用户是否取消了操作
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException("User cancelled during entity summary")

    # 8. Get summary description an LLM usage status
    # 调用智能摘要函数（与重建函数相同）
    description, llm_was_used = await _handle_entity_relation_summary(
        "Entity",
        entity_name,
        description_list,
        GRAPH_FIELD_SEP,
        global_config,
        llm_response_cache,
    )

    # 9. Build file_path within MAX_FILE_PATHS
    file_paths_list = []
    seen_paths = set()
    # 标记是否已存在占位符
    has_placeholder = False  # Indicating file_path has been truncated before
    # 获取配置
    max_file_paths = global_config.get("max_file_paths", DEFAULT_MAX_FILE_PATHS)
    file_path_placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )

    # Collect from already_file_paths, excluding placeholder
    # 从已有文件路径中收集（排除占位符）
    for fp in already_file_paths:
        # 检查是否是占位符
        if fp and fp.startswith(f"...{file_path_placeholder}"):  # Skip placeholders
            has_placeholder = True
            continue
        # 去重后添加
        if fp and fp not in seen_paths:
            file_paths_list.append(fp)
            seen_paths.add(fp)

    # Collect from new data
    for dp in nodes_data:
        file_path_item = dp.get("file_path")
        if file_path_item and file_path_item not in seen_paths:
            file_paths_list.append(file_path_item)
            seen_paths.add(file_path_item)

    # Apply count limit
    # 应用数量限制
    if len(file_paths_list) > max_file_paths:
        limit_method = global_config.get(
            "source_ids_limit_method", SOURCE_IDS_LIMIT_METHOD_KEEP
        )
        file_path_placeholder = global_config.get(
            "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
        )
        # Add + sign to indicate actual file count is higher
        # ⚠️ 关键：如果之前已有占位符，数量显示为"X+"
        original_count_str = (
            f"{len(file_paths_list)}+" if has_placeholder else str(len(file_paths_list))
        )

         # 根据策略截断
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
            file_paths_list.append(f"...{file_path_placeholder}...(FIFO)")
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]
            file_paths_list.append(f"...{file_path_placeholder}...(KEEP Old)")

        logger.info(
            f"Limited `{entity_name}`: file_path {original_count_str} -> {max_file_paths} ({limit_method})"
        )
    # Finalize file_path
    # 最终的file_path字符串
    file_path = GRAPH_FIELD_SEP.join(file_paths_list)

    # 10.Log based on actual LLM usage
    num_fragment = len(description_list)
    already_fragment = len(already_description)
    if llm_was_used:
        status_message = f"LLMmrg: `{entity_name}` | {already_fragment}+{num_fragment - already_fragment}"
    else:
        status_message = f"Merged: `{entity_name}` | {already_fragment}+{num_fragment - already_fragment}"

    truncation_info = truncation_info_log = ""
    if len(source_ids) < len(full_source_ids):
        # Add truncation info from apply_source_ids_limit if truncation occurred
        truncation_info_log = f"{limit_method} {len(source_ids)}/{len(full_source_ids)}"
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            truncation_info = truncation_info_log
        else:
            truncation_info = "KEEP Old"

    deduplicated_num = already_fragment + len(nodes_data) - num_fragment
    dd_message = ""
    if deduplicated_num > 0:
        # Duplicated description detected across multiple trucks for the same entity
        dd_message = f"dd {deduplicated_num}"

    if dd_message or truncation_info_log:
        status_message += (
            f" ({', '.join(filter(None, [truncation_info_log, dd_message]))})"
        )

    # Add message to pipeline satus when merge happens
    if already_fragment > 0 or llm_was_used:
        logger.info(status_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = status_message
                pipeline_status["history_messages"].append(status_message)
    else:
        logger.debug(status_message)

    # 11. Update both graph and vector db
    # 写入图数据库和向量库
    # 组装节点数据
    node_data = dict(
        entity_id=entity_name,
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        file_path=file_path,
        created_at=int(time.time()),
        truncate=truncation_info,
    )
    # 写入图数据库
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    # 添加entity_name字段（用于向量库）
    node_data["entity_name"] = entity_name
    # 更新向量库
    if entity_vdb is not None:
        # 计算向量库ID
        entity_vdb_id = compute_mdhash_id(str(entity_name), prefix="ent-")
        # 向量库content："{实体名}\n{描述}"
        entity_content = f"{entity_name}\n{description}"
        # 组装数据
        data_for_vdb = {
            entity_vdb_id: {
                "entity_name": entity_name,
                "entity_type": entity_type,
                "content": entity_content,
                "source_id": source_id,
                "file_path": file_path,
            }
        }
        # 使用安全包装器写入
        await safe_vdb_operation_with_exception(
            operation=lambda payload=data_for_vdb: entity_vdb.upsert(payload),
            operation_name="entity_upsert",
            entity_name=entity_name,
            max_retries=3,
            retry_delay=0.1,
        )
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,        # 源实体名称
    tgt_id: str,         # 目标实体名称
    edges_data: list[dict],     # 新提取的关系数据列表
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage | None,
    entity_vdb: BaseVectorStorage | None,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    added_entities: list = None,  # New parameter to track entities added during edge processing追踪边处理过程中创建的实体
    relation_chunks_storage: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
):
    """负责插入新文档时的关系合并逻辑，插入新文档时，将新提取的关系数据与已有数据合并"""
    # 自环检查：避免 "张三" -> "张三" 这种无意义关系
    if src_id == tgt_id:
        return None

    # 初始化存储已有数据的列表
    already_edge = None
    already_weights = []    # 已有的权重值
    already_source_ids = []     # 已有的chunk IDs
    already_description = []        # 已有的描述片段
    already_keywords = []       # 已有的关键词（可能是逗号分隔的字符串）
    already_file_paths = []      # 已有的文件路径

    # 1. Get existing edge data from graph storage
    # 从图数据库读取src_id和tgt_id已有关系，如果存在，记录下来元数据信息
    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        # Handle the case where get_edge returns None or missing fields
        # 如果关系存在，则提取相关的元数据信息到上面的列表，并做好缺失字段的容错处理
        if already_edge:
            # Get weight with default 1.0 if missing
            # 提取权重
            already_weights.append(already_edge.get("weight", 1.0))

            # Get source_id with empty string default if missing or None
            # 提取并分割source_id字段
            if already_edge.get("source_id") is not None:
                already_source_ids.extend(
                    already_edge["source_id"].split(GRAPH_FIELD_SEP)
                )

            # Get file_path with empty string default if missing or None
            # 提取文件路径、描述、关键词
            if already_edge.get("file_path") is not None:
                already_file_paths.extend(
                    already_edge["file_path"].split(GRAPH_FIELD_SEP)
                )

            # Get description with empty string default if missing or None
            if already_edge.get("description") is not None:
                already_description.extend(
                    already_edge["description"].split(GRAPH_FIELD_SEP)
                )

            # Get keywords with empty string default if missing or None
            if already_edge.get("keywords") is not None:
                already_keywords.extend(
                    split_string_by_multi_markers(
                        already_edge["keywords"], [GRAPH_FIELD_SEP]
                    )
                )
    # 提取传入的新数据的chunk IDs
    new_source_ids = [dp["source_id"] for dp in edges_data if dp.get("source_id")]
    # 构造存储键：例如 "张三<SEP>李四"（确定的一对 src_id, tgt_id 键值唯一）
    storage_key = make_relation_chunk_key(src_id, tgt_id)

    # 从 relation_chunks_storage 读取完整列表（老的）
    existing_full_source_ids = []
    if relation_chunks_storage is not None:
        # src_id, tgt_id获取的key是唯一的，通过读取来判断是否数据库里已经存在了这个关系的完整chunk列表
        stored_chunks = await relation_chunks_storage.get_by_id(storage_key)
        if stored_chunks and isinstance(stored_chunks, dict):
            # 获取已有的完整chunk ID列表
            existing_full_source_ids = [
                chunk_id for chunk_id in stored_chunks.get("chunk_ids", []) if chunk_id
            ]
    # 回退策略：如果relation_chunks_storage没有存储完整列表，从图数据库获取相应的id
    # already_source_ids是最开始那步从图数据库中获取元数据时得到了的东西
    if not existing_full_source_ids:
        existing_full_source_ids = [
            chunk_id for chunk_id in already_source_ids if chunk_id
        ]

    # 2. Merge new source ids with existing ones
    # 合并新旧chunk IDs（去重、排序）
    # 包含新旧、所有待处理的chunk id
    full_source_ids = merge_source_ids(existing_full_source_ids, new_source_ids)

    # 更新 relation_chunks_storage，保存完整列表
    if relation_chunks_storage is not None and full_source_ids:
        await relation_chunks_storage.upsert(
            {
                storage_key: {
                    "chunk_ids": full_source_ids,
                    "count": len(full_source_ids),
                }
            }
        )

    # 3. Finalize source_id by applying source ids limit
    # 获取限制策略和上限
    limit_method = global_config.get("source_ids_limit_method")
    max_source_limit = global_config.get("max_source_ids_per_relation")
    # 应用限制，截断chunk ID列表，后续操作更新只会影响这些chunk ID对应的数据
    source_ids = apply_source_ids_limit(
        full_source_ids,
        max_source_limit,
        limit_method,
        identifier=f"`{src_id}`~`{tgt_id}`",
    )
    limit_method = (
        global_config.get("source_ids_limit_method") or SOURCE_IDS_LIMIT_METHOD_KEEP
    )

    # 4. Only keep edges with source_id in the final source_ids list if in KEEP mode
    # KEEP策略：丢弃来自被截断chunk的新关系数据
    if limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP:
        # 只保留前30个chunk
        allowed_source_ids = set(source_ids)
        filtered_edges = []
        for dp in edges_data:
            source_id = dp.get("source_id")
            # Skip relationship fragments sourced from chunks dropped by keep oldest cap
            # 如果新关系来自chunk-31到chunk-80，且这些chunk不在原有列表中
            # 则丢弃这条关系数据
            if (
                source_id
                # 这个数据不在已经阶段了的chunk列表中
                and source_id not in allowed_source_ids
                # 且这个chunk不是不在relation_chunks_storage里src_id、tgt_id组成的key对应的chunk中已经存在的chunk
                and source_id not in existing_full_source_ids
            ):
                continue
            # 没有被continue的才保存下来
            filtered_edges.append(dp)
        edges_data = filtered_edges
    else:  # In FIFO mode, keep all edges - truncation happens at source_ids level only
        # FIFO策略：保留所有新数据，截断发生在source_ids层面
        edges_data = list(edges_data)

    # 5. Check if we need to skip summary due to source_ids limit
    # 特殊情况：KEEP策略 + 已满 + 无新数据可用
    # 如果实体已满50个chunk（有50个chunk都包含这个实体的信息），且使用KEEP策略
    # 新文档的信息会被完全忽略
    # 避免浪费资源进行无效的LLM调用
    if (
        limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP
        and len(existing_full_source_ids) >= max_source_limit
        and not edges_data
    ):
        if already_edge:
            # 直接返回已有数据，跳过后续处理
            logger.info(
                f"Skipped `{src_id}`~`{tgt_id}`: KEEP old chunks  {already_source_ids}/{len(full_source_ids)}"
            )
            # 直接返回已有节点数据
            existing_edge_data = dict(already_edge)
            return existing_edge_data
        else:
            logger.error(
                f"Internal Error: already_node missing for `{src_id}`~`{tgt_id}`"
            )
            raise ValueError(
                f"Internal Error: already_node missing for `{src_id}`~`{tgt_id}`"
            )

    # 6.1 Finalize source_id
    # 最终的source_id字段（用于图数据库）
    source_id = GRAPH_FIELD_SEP.join(source_ids)

    # 6.2 Finalize weight by summing new edges and existing weights
    # 累加权重
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)

    # 6.2 Finalize keywords by merging existing and new keywords
    # 合并关键词
    all_keywords = set()
    # Process already_keywords (which are comma-separated)
    # 处理已有关键词
    for keyword_str in already_keywords:    #already_keywords代表传入的tgt_id、src_id在图数据库中已经存在的关系的keywords字段，可能是逗号分隔的字符串
        if keyword_str:  # Skip empty strings
            all_keywords.update(k.strip() for k in keyword_str.split(",") if k.strip())
    # Process new keywords from edges_data
    # 处理新关键词
    for edge in edges_data:
        if edge.get("keywords"):
            all_keywords.update(
                k.strip() for k in edge["keywords"].split(",") if k.strip()
            )
    # Join all unique keywords with commas
    # 排序后合并：{"同事", "朋友", "合作伙伴"} -> "合作伙伴, 同事, 朋友"
    keywords = ",".join(sorted(all_keywords))

    # 7. Deduplicate by description, keeping first occurrence in the same document
    # 处理description
    unique_edges = {} #保存的是一个description到完整数据的映射
    for dp in edges_data:
        description_value = dp.get("description")
        # 如果不存在description，则跳过
        if not description_value:
            continue
        # 如果描述重复，跳过
        if description_value not in unique_edges:
            unique_edges[description_value] = dp

    # Sort description by timestamp, then by description length (largest to smallest) when timestamps are the same
    # # 排序：先按时间戳，再按描述长度（降序）对unique_edges进行排序
    sorted_edges = sorted(
        unique_edges.values(),
        key=lambda x: (x.get("timestamp", 0), -len(x.get("description", ""))),
    )
    # 将对应边的描述信息展开为一个list
    sorted_descriptions = [dp["description"] for dp in sorted_edges]

    # Combine already_description with sorted new descriptions
    # 合并新旧描述
    description_list = already_description + sorted_descriptions
    if not description_list:
        logger.error(f"Relation {src_id}~{tgt_id} has no description")
        raise ValueError(f"Relation {src_id}~{tgt_id} has no description")

    # Check for cancellation before LLM summary
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException(
                    "User cancelled during relation summary"
                )

    # 8. Get summary description an LLM usage status
    # 调用大模型进行智能摘要
    description, llm_was_used = await _handle_entity_relation_summary(
        "Relation",
        f"({src_id}, {tgt_id})",
        description_list,
        GRAPH_FIELD_SEP,
        global_config,
        llm_response_cache,
    )

    # 9. Build file_path within MAX_FILE_PATHS limit
    # 文件路径处理
    file_paths_list = []
    # 具体的单个路径
    seen_paths = set()
    # 标记是否已有占位符
    has_placeholder = False  # Track if already_file_paths contains placeholder

    max_file_paths = global_config.get("max_file_paths", DEFAULT_MAX_FILE_PATHS)
    file_path_placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )

    # Collect from already_file_paths, excluding placeholder
    # 收集已有文件路径
    for fp in already_file_paths:
        # Check if this is a placeholder record
        # 检查占位符
        if fp and fp.startswith(f"...{file_path_placeholder}"):  # Skip placeholders
            has_placeholder = True
            continue
        if fp and fp not in seen_paths:
            file_paths_list.append(fp)
            seen_paths.add(fp)

    # Collect from new data
    # 收集新文件路径
    for dp in edges_data:
        file_path_item = dp.get("file_path")
        if file_path_item and file_path_item not in seen_paths:
            file_paths_list.append(file_path_item)
            seen_paths.add(file_path_item)

    # Apply count limit
    max_file_paths = global_config.get("max_file_paths")

    # 应用数量限制
    if len(file_paths_list) > max_file_paths:
        limit_method = global_config.get(
            "source_ids_limit_method", SOURCE_IDS_LIMIT_METHOD_KEEP
        )
        file_path_placeholder = global_config.get(
            "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
        )

        # Add + sign to indicate actual file count is higher
        original_count_str = (
            f"{len(file_paths_list)}+" if has_placeholder else str(len(file_paths_list))
        )

        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            # FIFO：保留尾部（最新的）
            file_paths_list = file_paths_list[-max_file_paths:]
            file_paths_list.append(f"...{file_path_placeholder}...(FIFO)")
        else:
            # KEEP: keep head (earliest), discard tail
            # KEEP：保留头部（最早的）
            file_paths_list = file_paths_list[:max_file_paths]
            file_paths_list.append(f"...{file_path_placeholder}...(KEEP Old)")

        logger.info(
            f"Limited `{src_id}`~`{tgt_id}`: file_path {original_count_str} -> {max_file_paths} ({limit_method})"
        )
    # Finalize file_path
    # 最终的file_path字段
    # 这是一个list，每个元素是一个文件路径，最后通过GRAPH_FIELD_SEP连接成字符串存储在图数据库中
    # 一条边可能有多个文件
    file_path = GRAPH_FIELD_SEP.join(file_paths_list)

    # 10. Log based on actual LLM usage
    num_fragment = len(description_list)
    already_fragment = len(already_description)

    # 根据是否使用LLM生成不同的日志前缀
    if llm_was_used:
        status_message = f"LLMmrg: `{src_id}`~`{tgt_id}` | {already_fragment}+{num_fragment - already_fragment}"
    else:
        status_message = f"Merged: `{src_id}`~`{tgt_id}` | {already_fragment}+{num_fragment - already_fragment}"

    # 添加截断信息
    truncation_info = truncation_info_log = ""
    if len(source_ids) < len(full_source_ids):
        # Add truncation info from apply_source_ids_limit if truncation occurred
        truncation_info_log = f"{limit_method} {len(source_ids)}/{len(full_source_ids)}"
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            truncation_info = truncation_info_log
        else:
            truncation_info = "KEEP Old"
    # 添加去重信息
    deduplicated_num = already_fragment + len(edges_data) - num_fragment
    dd_message = ""
    if deduplicated_num > 0:
        # Duplicated description detected across multiple trucks for the same entity
        dd_message = f"dd {deduplicated_num}"
    # 组合消息
    if dd_message or truncation_info_log:
        status_message += (
            f" ({', '.join(filter(None, [truncation_info_log, dd_message]))})"
        )

    # Add message to pipeline satus when merge happens
    # 记录日志和更新管道状态
    if already_fragment > 0 or llm_was_used:
        logger.info(status_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = status_message
                pipeline_status["history_messages"].append(status_message)
    else:
        logger.debug(status_message)

    # 11. Update both graph and vector db
    # 遍历源节点和目标节点
    for need_insert_id in [src_id, tgt_id]:
        # Optimization: Use get_node instead of has_node + get_node
        existing_node = await knowledge_graph_inst.get_node(need_insert_id)

        if existing_node is None:
            # Node doesn't exist - create new node
            # 节点（起点、终点）不存在，创建新节点
            node_created_at = int(time.time())
            node_data = {
                "entity_id": need_insert_id,
                "source_id": source_id,
                "description": description,
                "entity_type": "UNKNOWN",
                "file_path": file_path,
                "created_at": node_created_at,
                "truncate": "",
            }
            await knowledge_graph_inst.upsert_node(need_insert_id, node_data=node_data)

            # Update entity_chunks_storage for the newly created entity
            # 更新 entity_chunks_storage
            if entity_chunks_storage is not None:
                chunk_ids = [chunk_id for chunk_id in full_source_ids if chunk_id]
                if chunk_ids:
                    await entity_chunks_storage.upsert(
                        {
                            need_insert_id: {
                                "chunk_ids": chunk_ids,
                                "count": len(chunk_ids),
                            }
                        }
                    )
            # 更新向量库
            if entity_vdb is not None:
                entity_vdb_id = compute_mdhash_id(need_insert_id, prefix="ent-")
                entity_content = f"{need_insert_id}\n{description}"
                vdb_data = {
                    entity_vdb_id: {
                        "content": entity_content,
                        "entity_name": need_insert_id,
                        "source_id": source_id,
                        "entity_type": "UNKNOWN",
                        "file_path": file_path,
                    }
                }
                await safe_vdb_operation_with_exception(
                    operation=lambda payload=vdb_data: entity_vdb.upsert(payload),
                    operation_name="added_entity_upsert",
                    entity_name=need_insert_id,
                    max_retries=3,
                    retry_delay=0.1,
                )

            # Track entities added during edge processing
            # 追踪新创建的实体
            if added_entities is not None:
                entity_data = {
                    "entity_name": need_insert_id,
                    "entity_type": "UNKNOWN",
                    "description": description,
                    "source_id": source_id,
                    "file_path": file_path,
                    "created_at": node_created_at,
                }
                added_entities.append(entity_data)
        else:
            # Node exists - update its source_ids by merging with new source_ids
            # 节点已存在，更新source_ids
            updated = False  # Track if any update occurred

            # 1. Get existing full source_ids from entity_chunks_storage
            # 获取完整chunk列表
            existing_full_source_ids = []
            if entity_chunks_storage is not None:
                # 获取对应起点、终点的chunk信息
                stored_chunks = await entity_chunks_storage.get_by_id(need_insert_id)
                if stored_chunks and isinstance(stored_chunks, dict):
                    existing_full_source_ids = [
                        chunk_id
                        for chunk_id in stored_chunks.get("chunk_ids", [])
                        if chunk_id
                    ]

            # If not in entity_chunks_storage, get from graph database
            # 回退：从图数据库获取chunk ID列表，如果entity_chunks_storage没有记录
            if not existing_full_source_ids:
                if existing_node.get("source_id"):
                    existing_full_source_ids = existing_node["source_id"].split(
                        GRAPH_FIELD_SEP
                    )

            # 2. Merge with new source_ids from this relationship
            # 合并chunk IDs
            new_source_ids_from_relation = [
                chunk_id for chunk_id in source_ids if chunk_id
            ]
            merged_full_source_ids = merge_source_ids(
                existing_full_source_ids, new_source_ids_from_relation
            )

            # 3. Save merged full list to entity_chunks_storage (conditional)
            # 更新 entity_chunks_storage（条件：有变化）
            if (
                entity_chunks_storage is not None
                and merged_full_source_ids != existing_full_source_ids
            ):
                updated = True
                await entity_chunks_storage.upsert(
                    {
                        need_insert_id: {
                            "chunk_ids": merged_full_source_ids,
                            "count": len(merged_full_source_ids),
                        }
                    }
                )

            # 4. Apply source_ids limit for graph and vector db
            limit_method = global_config.get(
                "source_ids_limit_method", SOURCE_IDS_LIMIT_METHOD_KEEP
            )
            max_source_limit = global_config.get("max_source_ids_per_entity")
            limited_source_ids = apply_source_ids_limit(
                merged_full_source_ids,
                max_source_limit,
                limit_method,
                identifier=f"`{need_insert_id}`",
            )

            # 5. Update graph database and vector database with limited source_ids (conditional)
            # 更新图数据库和向量库（条件：source_id有变化）
            limited_source_id_str = GRAPH_FIELD_SEP.join(limited_source_ids)

            if limited_source_id_str != existing_node.get("source_id", ""):
                updated = True
                updated_node_data = {
                    **existing_node,
                    "source_id": limited_source_id_str,
                }
                await knowledge_graph_inst.upsert_node(
                    need_insert_id, node_data=updated_node_data
                )

                # Update vector database
                # 更新向量库
                if entity_vdb is not None:
                    entity_vdb_id = compute_mdhash_id(need_insert_id, prefix="ent-")
                    entity_content = (
                        f"{need_insert_id}\n{existing_node.get('description', '')}"
                    )
                    vdb_data = {
                        entity_vdb_id: {
                            "content": entity_content,
                            "entity_name": need_insert_id,
                            "source_id": limited_source_id_str,
                            "entity_type": existing_node.get("entity_type", "UNKNOWN"),
                            "file_path": existing_node.get(
                                "file_path", "unknown_source"
                            ),
                        }
                    }
                    await safe_vdb_operation_with_exception(
                        operation=lambda payload=vdb_data: entity_vdb.upsert(payload),
                        operation_name="existing_entity_update",
                        entity_name=need_insert_id,
                        max_retries=3,
                        retry_delay=0.1,
                    )

            # 6. Log once at the end if any update occurred
            if updated:
                status_message = f"Chunks appended from relation: `{need_insert_id}`"
                logger.info(status_message)
                if pipeline_status is not None and pipeline_status_lock is not None:
                    async with pipeline_status_lock:
                        pipeline_status["latest_message"] = status_message
                        pipeline_status["history_messages"].append(status_message)
    # 写入图数据库
    edge_created_at = int(time.time())
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
            file_path=file_path,
            created_at=edge_created_at,
            truncate=truncation_info,
        ),
    )
    # 准备返回数据
    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
        source_id=source_id,
        file_path=file_path,
        created_at=edge_created_at,
        truncate=truncation_info,
        weight=weight,
    )

    # Sort src_id and tgt_id to ensure consistent ordering (smaller string first)
    # 排序确保一致性
    if src_id > tgt_id:
        src_id, tgt_id = tgt_id, src_id
    # 更新关系的向量库
    if relationships_vdb is not None:
        # 计算ID（正向和反向）
        rel_vdb_id = compute_mdhash_id(src_id + tgt_id, prefix="rel-")
        rel_vdb_id_reverse = compute_mdhash_id(tgt_id + src_id, prefix="rel-")
        # 删除旧记录（处理方向变化）
        try:
            await relationships_vdb.delete([rel_vdb_id, rel_vdb_id_reverse])
        except Exception as e:
            logger.debug(
                f"Could not delete old relationship vector records {rel_vdb_id}, {rel_vdb_id_reverse}: {e}"
            )
        # 插入新纪录
        rel_content = f"{keywords}\t{src_id}\n{tgt_id}\n{description}"
        vdb_data = {
            rel_vdb_id: {
                "src_id": src_id,
                "tgt_id": tgt_id,
                "source_id": source_id,
                "content": rel_content,
                "keywords": keywords,
                "description": description,
                "weight": weight,
                "file_path": file_path,
            }
        }
        await safe_vdb_operation_with_exception(
            operation=lambda payload=vdb_data: relationships_vdb.upsert(payload),
            operation_name="relationship_upsert",
            entity_name=f"{src_id}-{tgt_id}",
            max_retries=3,
            retry_delay=0.2,
        )

    return edge_data


async def merge_nodes_and_edges(
    chunk_results: list,        # 所有chunk的提取结果列表
    knowledge_graph_inst: BaseGraphStorage,     # 图数据库实例
    entity_vdb: BaseVectorStorage,      # 实体向量库
    relationships_vdb: BaseVectorStorage,       # 关系向量库
    global_config: dict[str, str],
    full_entities_storage: BaseKVStorage = None,        # 文档-实体映射存储
    full_relations_storage: BaseKVStorage = None,       # 文档-关系映射存储
    doc_id: str = None,         # 文档ID
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,    # LLM响应缓存
    entity_chunks_storage: BaseKVStorage | None = None,
    relation_chunks_storage: BaseKVStorage | None = None,
    current_file_number: int = 0,
    total_files: int = 0,
    file_path: str = "unknown_source",
) -> None:
    """文档插入流程的核心协调函数,负责将所有chunk的提取结果合并到知识图谱中。
    Two-phase merge: process all entities first, then all relationships

    This approach ensures data consistency by:
    1. Phase 1: Process all entities concurrently
    2. Phase 2: Process all relationships concurrently (may add missing entities)
    3. Phase 3: Update full_entities and full_relations storage with final results

    Args:
        chunk_results: List of tuples (maybe_nodes, maybe_edges) containing extracted entities and relationships
        knowledge_graph_inst: Knowledge graph storage
        entity_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        global_config: Global configuration
        full_entities_storage: Storage for document entity lists
        full_relations_storage: Storage for document relation lists
        doc_id: Document ID for storage indexing
        pipeline_status: Pipeline status dictionary
        pipeline_status_lock: Lock for pipeline status
        llm_response_cache: LLM response cache
        entity_chunks_storage: Storage tracking full chunk lists per entity
        relation_chunks_storage: Storage tracking full chunk lists per relation
        current_file_number: Current file number for logging
        total_files: Total files for logging
        file_path: File path for logging
    """

    # Check for cancellation at the start of merge
    # 取消检查：在合并开始前检查用户是否取消
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException("User cancelled during merge phase")

    # Collect all nodes and edges from all chunks
    # 初始化字典，用于收集所有chunk的提取结果包含的实体和关系
    all_nodes = defaultdict(list)       # {实体名: [实体数据列表]}
    all_edges = defaultdict(list)       # {(src,tgt): [关系数据列表]}

# chunk_results的结构：
# [
#     (maybe_nodes, maybe_edges),  # chunk-1的提取结果
#     (maybe_nodes, maybe_edges),  # chunk-2的提取结果
#     ...
# ]
# 其中 maybe_nodes = {"张三": [entity_data1, entity_data2], ...}
#      maybe_edges = {("张三","李四"): [edge_data1, edge_data2], ...}

    for maybe_nodes, maybe_edges in chunk_results:
        # Collect nodes
        # 收集实体：按实体名分组
        for entity_name, entities in maybe_nodes.items():
            all_nodes[entity_name].extend(entities)

        # Collect edges with sorted keys for undirected graph
        # 收集关系：对边的键进行排序，确保无向图的一致性
        for edge_key, edges in maybe_edges.items():
            # 排序：("李四","张三") -> ("张三","李四")
            sorted_edge_key = tuple(sorted(edge_key))
            # 填入字典
            all_edges[sorted_edge_key].extend(edges)

    # 统计总数
    total_entities_count = len(all_nodes)
    total_relations_count = len(all_edges)

    log_message = f"Merging stage {current_file_number}/{total_files}: {file_path}"
    logger.info(log_message)
    # 更新管道状态（前端可以实时看到进度）
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

    # Get max async tasks limit from global_config for semaphore control
    # 获取LLM并发数，乘以2作为图操作的并发数
    # 原因：图操作通常比LLM调用快，可以更高并发
    graph_max_async = global_config.get("llm_model_max_async", 4) * 2
    # 创建信号量限制同时运行的任务数
    semaphore = asyncio.Semaphore(graph_max_async)

    # ===== Phase 1: Process all entities concurrently =====
    # 并发处理所有实体
    log_message = f"Phase 1: Processing {total_entities_count} entities from {doc_id} (async: {graph_max_async})"
    logger.info(log_message)
    # 更新管道状态（前端可以实时看到进度）
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

    # 定义单个实体处理函数（带锁和信号量控制）
    async def _locked_process_entity_name(entity_name, entities):
        # 信号量控制：同时最多graph_max_async个任务运行
        async with semaphore:
            # Check for cancellation before processing entity
            # 检查取消
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    if pipeline_status.get("cancellation_requested", False):
                        raise PipelineCancelledException(
                            "User cancelled during entity merge"
                        )
            # 获取工作空间配置
            workspace = global_config.get("workspace", "")
            # 构造命名空间
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            # 获取分布式锁：确保同一实体在多进程/多线程中只被一个处理
            async with get_storage_keyed_lock(
                [entity_name], namespace=namespace, enable_logging=False
            ):
                try:
                    logger.debug(f"Processing entity {entity_name}")
                    # 调用核心合并函数
                    entity_data = await _merge_nodes_then_upsert(
                        entity_name,
                        entities,
                        knowledge_graph_inst,
                        entity_vdb,
                        global_config,
                        pipeline_status,
                        pipeline_status_lock,
                        llm_response_cache,
                        entity_chunks_storage,
                    )

                    return entity_data

                except Exception as e:
                    error_msg = f"Error processing entity `{entity_name}`: {e}"
                    logger.error(error_msg)

                    # Try to update pipeline status, but don't let status update failure affect main exception
                    # 尝试更新状态，但不让状态更新失败影响主异常
                    try:
                        if (
                            pipeline_status is not None
                            and pipeline_status_lock is not None
                        ):
                            async with pipeline_status_lock:
                                pipeline_status["latest_message"] = error_msg
                                pipeline_status["history_messages"].append(error_msg)
                    except Exception as status_error:
                        logger.error(
                            f"Failed to update pipeline status: {status_error}"
                        )

                    # Re-raise the original exception with a prefix
                    # 重新抛出异常，添加实体名前缀
                    prefixed_exception = create_prefixed_exception(
                        e, f"`{entity_name}`"
                    )
                    raise prefixed_exception from e

    # Create entity processing tasks
    # 为每个实体创建异步任务
    entity_tasks = []
    for entity_name, entities in all_nodes.items():
        task = asyncio.create_task(_locked_process_entity_name(entity_name, entities))
        entity_tasks.append(task)

    # Execute entity tasks with error handling
    # 执行任务，遇到第一个异常时立即返回
    processed_entities = []
    if entity_tasks:
        done, pending = await asyncio.wait(
            entity_tasks, return_when=asyncio.FIRST_EXCEPTION   # 快速失败策略
        )

        first_exception = None
        processed_entities = []

        # 处理已完成的任务
        for task in done:
            try:
                # 获取任务结果或异常
                result = task.result()
            except BaseException as e:
                if first_exception is None:
                    first_exception = e      # 记录第一个异常
            else:
                processed_entities.append(result)   # 收集成功结果

        # 如果有异常，取消所有待处理任务
        if pending:
            for task in pending:
                task.cancel()   # 发送取消信号
            # 等待所有任务结束（可能返回取消异常）
            pending_results = await asyncio.gather(*pending, return_exceptions=True)
            for result in pending_results:
                if isinstance(result, BaseException):
                    if first_exception is None:
                        first_exception = result
                else:
                    processed_entities.append(result)
        # 如果有异常，抛出
        if first_exception is not None:
            raise first_exception

    # ===== Phase 2: Process all relationships concurrently =====
    # 并发处理所有关系
    log_message = f"Phase 2: Processing {total_relations_count} relations from {doc_id} (async: {graph_max_async})"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)
    # 定义关系处理函数
    async def _locked_process_edges(edge_key, edges):
        async with semaphore:
            # Check for cancellation before processing edges
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    if pipeline_status.get("cancellation_requested", False):
                        raise PipelineCancelledException(
                            "User cancelled during relation merge"
                        )

            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            # 排序边的端点以获取一致的锁顺序（避免死锁）
            sorted_edge_key = sorted([edge_key[0], edge_key[1]])
            # 对边的两个端点加锁
            async with get_storage_keyed_lock(
                sorted_edge_key,
                namespace=namespace,
                enable_logging=False,
            ):
                try:
                    # 跟踪在边处理过程中添加的实体
                    added_entities = []  # Track entities added during edge processing

                    logger.debug(f"Processing relation {sorted_edge_key}")
                    edge_data = await _merge_edges_then_upsert(
                        edge_key[0],
                        edge_key[1],
                        edges,
                        knowledge_graph_inst,
                        relationships_vdb,
                        entity_vdb,
                        global_config,
                        pipeline_status,
                        pipeline_status_lock,
                        llm_response_cache,
                        added_entities,  # 传入列表收集新增实体
                        relation_chunks_storage,
                        entity_chunks_storage,  # Add entity_chunks_storage parameter
                    )

                    if edge_data is None:
                        return None, []

                    return edge_data, added_entities

                except Exception as e:
                    error_msg = f"Error processing relation `{sorted_edge_key}`: {e}"
                    logger.error(error_msg)

                    # Try to update pipeline status, but don't let status update failure affect main exception
                    try:
                        if (
                            pipeline_status is not None
                            and pipeline_status_lock is not None
                        ):
                            async with pipeline_status_lock:
                                pipeline_status["latest_message"] = error_msg
                                pipeline_status["history_messages"].append(error_msg)
                    except Exception as status_error:
                        logger.error(
                            f"Failed to update pipeline status: {status_error}"
                        )

                    # Re-raise the original exception with a prefix
                    prefixed_exception = create_prefixed_exception(
                        e, f"{sorted_edge_key}"
                    )
                    raise prefixed_exception from e

    # Create relationship processing tasks
    edge_tasks = []
    for edge_key, edges in all_edges.items():
        task = asyncio.create_task(_locked_process_edges(edge_key, edges))
        edge_tasks.append(task)

    # Execute relationship tasks with error handling
    processed_edges = []
    # 收集所有在关系处理中新增的实体
    all_added_entities = []

    if edge_tasks:
        done, pending = await asyncio.wait(
            edge_tasks, return_when=asyncio.FIRST_EXCEPTION
        )

        first_exception = None

        for task in done:
            try:
                edge_data, added_entities = task.result()
            except BaseException as e:
                if first_exception is None:
                    first_exception = e
            else:
                if edge_data is not None:
                    processed_edges.append(edge_data)
                # 累积新增实体
                all_added_entities.extend(added_entities)

        if pending:
            for task in pending:
                task.cancel()
            pending_results = await asyncio.gather(*pending, return_exceptions=True)
            for result in pending_results:
                if isinstance(result, BaseException):
                    if first_exception is None:
                        first_exception = result
                else:
                    edge_data, added_entities = result
                    if edge_data is not None:
                        processed_edges.append(edge_data)
                    all_added_entities.extend(added_entities)

        if first_exception is not None:
            raise first_exception

    # ===== Phase 3: Update full_entities and full_relations storage =====
    # 更新在kv存储中的文档-实体和文档-关系索引
    if full_entities_storage and full_relations_storage and doc_id:
        try:
            # Merge all entities: original entities + entities added during edge processing
            # 合并所有实体：原始实体 + 边处理时新增的实体
            final_entity_names = set()

            # Add original processed entities
            # 添加Phase 1处理的实体
            for entity_data in processed_entities:
                if entity_data and entity_data.get("entity_name"):
                    final_entity_names.add(entity_data["entity_name"])

            # Add entities that were added during relationship processing
            # 添加Phase 2中新增的实体
            for added_entity in all_added_entities:
                if added_entity and added_entity.get("entity_name"):
                    final_entity_names.add(added_entity["entity_name"])

            # Collect all relation pairs
            # 收集所有关系对
            final_relation_pairs = set()
            for edge_data in processed_edges:
                if edge_data:
                    src_id = edge_data.get("src_id")
                    tgt_id = edge_data.get("tgt_id")
                    if src_id and tgt_id:
                        # 排序确保一致性
                        relation_pair = tuple(sorted([src_id, tgt_id]))
                        final_relation_pairs.add(relation_pair)

            log_message = f"Phase 3: Updating final {len(final_entity_names)}({len(processed_entities)}+{len(all_added_entities)}) entities and  {len(final_relation_pairs)} relations from {doc_id}"
            logger.info(log_message)
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

            # Update storage
            # 更新存储：文档ID -> 实体列表
            if final_entity_names:
                await full_entities_storage.upsert(
                    {
                        doc_id: {
                            "entity_names": list(final_entity_names),
                            "count": len(final_entity_names),
                        }
                    }
                )
            # 更新存储：文档ID -> 关系列表
            if final_relation_pairs:
                await full_relations_storage.upsert(
                    {
                        doc_id: {
                            "relation_pairs": [
                                list(pair) for pair in final_relation_pairs
                            ],
                            "count": len(final_relation_pairs),
                        }
                    }
                )

            logger.debug(
                f"Updated entity-relation index for document {doc_id}: {len(final_entity_names)} entities (original: {len(processed_entities)}, added: {len(all_added_entities)}), {len(final_relation_pairs)} relations"
            )

        except Exception as e:
            logger.error(
                f"Failed to update entity-relation index for document {doc_id}: {e}"
            )
            # Don't raise exception to avoid affecting main flow

    log_message = f"Completed merging: {len(processed_entities)} entities, {len(all_added_entities)} extra entities, {len(processed_edges)} relations"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)


async def extract_entities(
    chunks: dict[str, TextChunkSchema], # 文档分块字典
    global_config: dict[str, str],  # 全局配置
    pipeline_status: dict = None,    # 管道状态（用于前端进度显示）
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,    # LLM响应缓存
    text_chunks_storage: BaseKVStorage | None = None,   # chunk存储（用于记录缓存键）
) -> list:
    """"extract_entities 是文档插入流程的第一阶段，它并发地对所有文本chunk调用LLM进行实体和关系提取，
    并返回结构化结果供后续的 merge_nodes_and_edges 函数使用。"""
    # Check for cancellation at the start of entity extraction
    # 检查用户是否在提取开始前就取消了操作
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException(
                    "User cancelled during entity extraction"
                )

    use_llm_func: callable = global_config["llm_model_func"]
    # gleaning是二次提取策略：第一次提取后，再问LLM"还有遗漏的实体吗？"
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    # 将chunks字典转换为有序列表
    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get("language", DEFAULT_SUMMARY_LANGUAGE)
    # 获取要提取的实体类型（如：PERSON, ORGANIZATION, LOCATION等）
    entity_types = global_config["addon_params"].get(
        "entity_types", DEFAULT_ENTITY_TYPES
    )
    # 获取示例prompt
    examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=", ".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    processed_chunks = 0
    total_chunks = len(ordered_chunks)

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """Process a single chunk
        Args:
            chunk_key_dp (tuple[str, TextChunkSchema]):
                ("chunk-xxxxxx", {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int})
        Returns:
            tuple: (maybe_nodes, maybe_edges) containing extracted entities and relationships
        """
        nonlocal processed_chunks
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        # Get file path from chunk data or use default
        file_path = chunk_dp.get("file_path", "unknown_source")

        # Create cache keys collector for batch processing
        cache_keys_collector = []

        # Get initial extraction
        # Format system prompt without input_text for each chunk (enables OpenAI prompt caching across chunks)
        entity_extraction_system_prompt = PROMPTS[
            "entity_extraction_system_prompt"
        ].format(**context_base)
        # Format user prompts with input_text for each chunk
        entity_extraction_user_prompt = PROMPTS["entity_extraction_user_prompt"].format(
            **{**context_base, "input_text": content}
        )
        entity_continue_extraction_user_prompt = PROMPTS[
            "entity_continue_extraction_user_prompt"
        ].format(**{**context_base, "input_text": content})

        final_result, timestamp = await use_llm_func_with_cache(
            entity_extraction_user_prompt,
            use_llm_func,
            system_prompt=entity_extraction_system_prompt,
            llm_response_cache=llm_response_cache,
            cache_type="extract",
            chunk_id=chunk_key,
            cache_keys_collector=cache_keys_collector,
        )

        history = pack_user_ass_to_openai_messages(
            entity_extraction_user_prompt, final_result
        )

        # Process initial extraction with file path
        maybe_nodes, maybe_edges = await _process_extraction_result(
            final_result,
            chunk_key,
            timestamp,
            file_path,
            tuple_delimiter=context_base["tuple_delimiter"],
            completion_delimiter=context_base["completion_delimiter"],
        )

        # Process additional gleaning results only 1 time when entity_extract_max_gleaning is greater than zero.
        if entity_extract_max_gleaning > 0:
            glean_result, timestamp = await use_llm_func_with_cache(
                entity_continue_extraction_user_prompt,
                use_llm_func,
                system_prompt=entity_extraction_system_prompt,
                llm_response_cache=llm_response_cache,
                history_messages=history,
                cache_type="extract",
                chunk_id=chunk_key,
                cache_keys_collector=cache_keys_collector,
            )

            # Process gleaning result separately with file path
            glean_nodes, glean_edges = await _process_extraction_result(
                glean_result,
                chunk_key,
                timestamp,
                file_path,
                tuple_delimiter=context_base["tuple_delimiter"],
                completion_delimiter=context_base["completion_delimiter"],
            )

            # Merge results - compare description lengths to choose better version
            for entity_name, glean_entities in glean_nodes.items():
                if entity_name in maybe_nodes:
                    # Compare description lengths and keep the better one
                    original_desc_len = len(
                        maybe_nodes[entity_name][0].get("description", "") or ""
                    )
                    glean_desc_len = len(glean_entities[0].get("description", "") or "")

                    if glean_desc_len > original_desc_len:
                        maybe_nodes[entity_name] = list(glean_entities)
                    # Otherwise keep original version
                else:
                    # New entity from gleaning stage
                    maybe_nodes[entity_name] = list(glean_entities)

            for edge_key, glean_edges in glean_edges.items():
                if edge_key in maybe_edges:
                    # Compare description lengths and keep the better one
                    original_desc_len = len(
                        maybe_edges[edge_key][0].get("description", "") or ""
                    )
                    glean_desc_len = len(glean_edges[0].get("description", "") or "")

                    if glean_desc_len > original_desc_len:
                        maybe_edges[edge_key] = list(glean_edges)
                    # Otherwise keep original version
                else:
                    # New edge from gleaning stage
                    maybe_edges[edge_key] = list(glean_edges)

        # Batch update chunk's llm_cache_list with all collected cache keys
        if cache_keys_collector and text_chunks_storage:
            await update_chunk_cache_list(
                chunk_key,
                text_chunks_storage,
                cache_keys_collector,
                "entity_extraction",
            )

        processed_chunks += 1
        entities_count = len(maybe_nodes)
        relations_count = len(maybe_edges)
        log_message = f"Chunk {processed_chunks} of {total_chunks} extracted {entities_count} Ent + {relations_count} Rel {chunk_key}"
        logger.info(log_message)
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        # Return the extracted nodes and edges for centralized processing
        return maybe_nodes, maybe_edges

    # Get max async tasks limit from global_config
    chunk_max_async = global_config.get("llm_model_max_async", 4)
    semaphore = asyncio.Semaphore(chunk_max_async)

    async def _process_with_semaphore(chunk):
        async with semaphore:
            # Check for cancellation before processing chunk
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    if pipeline_status.get("cancellation_requested", False):
                        raise PipelineCancelledException(
                            "User cancelled during chunk processing"
                        )

            try:
                return await _process_single_content(chunk)
            except Exception as e:
                chunk_id = chunk[0]  # Extract chunk_id from chunk[0]
                prefixed_exception = create_prefixed_exception(e, chunk_id)
                raise prefixed_exception from e

    tasks = []
    for c in ordered_chunks:
        task = asyncio.create_task(_process_with_semaphore(c))
        tasks.append(task)

    # Wait for tasks to complete or for the first exception to occur
    # This allows us to cancel remaining tasks if any task fails
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    # Check if any task raised an exception and ensure all exceptions are retrieved
    first_exception = None
    chunk_results = []

    for task in done:
        try:
            exception = task.exception()
            if exception is not None:
                if first_exception is None:
                    first_exception = exception
            else:
                chunk_results.append(task.result())
        except Exception as e:
            if first_exception is None:
                first_exception = e

    # If any task failed, cancel all pending tasks and raise the first exception
    if first_exception is not None:
        # Cancel all pending tasks
        for pending_task in pending:
            pending_task.cancel()

        # Wait for cancellation to complete
        if pending:
            await asyncio.wait(pending)

        # Add progress prefix to the exception message
        progress_prefix = f"C[{processed_chunks + 1}/{total_chunks}]"

        # Re-raise the original exception with a prefix
        prefixed_exception = create_prefixed_exception(first_exception, progress_prefix)
        raise prefixed_exception from first_exception

    # If all tasks completed successfully, chunk_results already contains the results
    # Return the chunk_results for later processing in merge_nodes_and_edges
    return chunk_results


async def kg_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
    chunks_vdb: BaseVectorStorage = None,
) -> QueryResult | None:
    """
    Execute knowledge graph query and return unified QueryResult object.

    Args:
        query: Query string
        knowledge_graph_inst: Knowledge graph storage instance
        entities_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        text_chunks_db: Text chunks storage
        query_param: Query parameters
        global_config: Global configuration
        hashing_kv: Cache storage
        system_prompt: System prompt
        chunks_vdb: Document chunks vector database

    Returns:
        QueryResult | None: Unified query result object containing:
            - content: Non-streaming response text content
            - response_iterator: Streaming response iterator
            - raw_data: Complete structured data (including references and metadata)
            - is_streaming: Whether this is a streaming result

        Based on different query_param settings, different fields will be populated:
        - only_need_context=True: content contains context string
        - only_need_prompt=True: content contains complete prompt
        - stream=True: response_iterator contains streaming response, raw_data contains complete data
        - default: content contains LLM response text, raw_data contains complete data

        Returns None when no relevant context could be constructed for the query.
    """
    if not query:
        return QueryResult(content=PROMPTS["fail_response"])

    if query_param.model_func:
        use_model_func = query_param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    hl_keywords, ll_keywords = await get_keywords_from_query(
        query, query_param, global_config, hashing_kv
    )

    logger.debug(f"High-level keywords: {hl_keywords}")
    logger.debug(f"Low-level  keywords: {ll_keywords}")

    # Handle empty keywords
    if ll_keywords == [] and query_param.mode in ["local", "hybrid", "mix"]:
        logger.warning("low_level_keywords is empty")
    if hl_keywords == [] and query_param.mode in ["global", "hybrid", "mix"]:
        logger.warning("high_level_keywords is empty")
    if hl_keywords == [] and ll_keywords == []:
        if len(query) < 50:
            logger.warning(f"Forced low_level_keywords to origin query: {query}")
            ll_keywords = [query]
        else:
            return QueryResult(content=PROMPTS["fail_response"])

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    # Build query context (unified interface)
    context_result = await _build_query_context(
        query,
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        chunks_vdb,
    )

    if context_result is None:
        logger.info("[kg_query] No query context could be built; returning no-result.")
        return None

    # Return different content based on query parameters
    if query_param.only_need_context and not query_param.only_need_prompt:
        return QueryResult(
            content=context_result.context, raw_data=context_result.raw_data
        )

    user_prompt = f"\n\n{query_param.user_prompt}" if query_param.user_prompt else "n/a"
    response_type = (
        query_param.response_type
        if query_param.response_type
        else "Multiple Paragraphs"
    )

    # Build system prompt
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        response_type=response_type,
        user_prompt=user_prompt,
        context_data=context_result.context,
    )

    user_query = query

    if query_param.only_need_prompt:
        prompt_content = "\n\n".join([sys_prompt, "---User Query---", user_query])
        return QueryResult(content=prompt_content, raw_data=context_result.raw_data)

    # Call LLM
    tokenizer: Tokenizer = global_config["tokenizer"]
    len_of_prompts = len(tokenizer.encode(query + sys_prompt))
    logger.debug(
        f"[kg_query] Sending to LLM: {len_of_prompts:,} tokens (Query: {len(tokenizer.encode(query))}, System: {len(tokenizer.encode(sys_prompt))})"
    )

    # Handle cache
    args_hash = compute_args_hash(
        query_param.mode,
        query,
        query_param.response_type,
        query_param.top_k,
        query_param.chunk_top_k,
        query_param.max_entity_tokens,
        query_param.max_relation_tokens,
        query_param.max_total_tokens,
        hl_keywords_str,
        ll_keywords_str,
        query_param.user_prompt or "",
        query_param.enable_rerank,
    )

    cached_result = await handle_cache(
        hashing_kv, args_hash, user_query, query_param.mode, cache_type="query"
    )

    if cached_result is not None:
        cached_response, _ = cached_result  # Extract content, ignore timestamp
        logger.info(
            " == LLM cache == Query cache hit, using cached response as query result"
        )
        response = cached_response
    else:
        response = await use_model_func(
            user_query,
            system_prompt=sys_prompt,
            history_messages=query_param.conversation_history,
            enable_cot=True,
            stream=query_param.stream,
        )

        if hashing_kv and hashing_kv.global_config.get("enable_llm_cache"):
            queryparam_dict = {
                "mode": query_param.mode,
                "response_type": query_param.response_type,
                "top_k": query_param.top_k,
                "chunk_top_k": query_param.chunk_top_k,
                "max_entity_tokens": query_param.max_entity_tokens,
                "max_relation_tokens": query_param.max_relation_tokens,
                "max_total_tokens": query_param.max_total_tokens,
                "hl_keywords": hl_keywords_str,
                "ll_keywords": ll_keywords_str,
                "user_prompt": query_param.user_prompt or "",
                "enable_rerank": query_param.enable_rerank,
            }
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=response,
                    prompt=query,
                    mode=query_param.mode,
                    cache_type="query",
                    queryparam=queryparam_dict,
                ),
            )

    # Return unified result based on actual response type
    if isinstance(response, str):
        # Non-streaming response (string)
        if len(response) > len(sys_prompt):
            response = (
                response.replace(sys_prompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .strip()
            )

        return QueryResult(content=response, raw_data=context_result.raw_data)
    else:
        # Streaming response (AsyncIterator)
        return QueryResult(
            response_iterator=response,
            raw_data=context_result.raw_data,
            is_streaming=True,
        )


async def get_keywords_from_query(
    query: str,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> tuple[list[str], list[str]]:
    """
    Retrieves high-level and low-level keywords for RAG operations.

    This function checks if keywords are already provided in query parameters,
    and if not, extracts them from the query text using LLM.

    Args:
        query: The user's query text
        query_param: Query parameters that may contain pre-defined keywords
        global_config: Global configuration dictionary
        hashing_kv: Optional key-value storage for caching results

    Returns:
        A tuple containing (high_level_keywords, low_level_keywords)
    """
    # Check if pre-defined keywords are already provided
    if query_param.hl_keywords or query_param.ll_keywords:
        return query_param.hl_keywords, query_param.ll_keywords

    # Extract keywords using extract_keywords_only function which already supports conversation history
    hl_keywords, ll_keywords = await extract_keywords_only(
        query, query_param, global_config, hashing_kv
    )
    return hl_keywords, ll_keywords


async def extract_keywords_only(
    text: str,
    param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> tuple[list[str], list[str]]:
    """
    Extract high-level and low-level keywords from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extracts keywords (hl_keywords, ll_keywords).
    """

    # 1. Build the examples
    examples = "\n".join(PROMPTS["keywords_extraction_examples"])

    language = global_config["addon_params"].get("language", DEFAULT_SUMMARY_LANGUAGE)

    # 2. Handle cache if needed - add cache type for keywords
    args_hash = compute_args_hash(
        param.mode,
        text,
        language,
    )
    cached_result = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="keywords"
    )
    if cached_result is not None:
        cached_response, _ = cached_result  # Extract content, ignore timestamp
        try:
            keywords_data = json_repair.loads(cached_response)
            return keywords_data.get("high_level_keywords", []), keywords_data.get(
                "low_level_keywords", []
            )
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for keywords, proceeding with extraction"
            )

    # 3. Build the keyword-extraction prompt
    kw_prompt = PROMPTS["keywords_extraction"].format(
        query=text,
        examples=examples,
        language=language,
    )

    tokenizer: Tokenizer = global_config["tokenizer"]
    len_of_prompts = len(tokenizer.encode(kw_prompt))
    logger.debug(
        f"[extract_keywords] Sending to LLM: {len_of_prompts:,} tokens (Prompt: {len_of_prompts})"
    )

    # 4. Call the LLM for keyword extraction
    if param.model_func:
        use_model_func = param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    result = await use_model_func(kw_prompt, keyword_extraction=True)

    # 5. Parse out JSON from the LLM response
    result = remove_think_tags(result)
    try:
        keywords_data = json_repair.loads(result)
        if not keywords_data:
            logger.error("No JSON-like structure found in the LLM respond.")
            return [], []
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"LLM respond: {result}")
        return [], []

    hl_keywords = keywords_data.get("high_level_keywords", [])
    ll_keywords = keywords_data.get("low_level_keywords", [])

    # 6. Cache only the processed keywords with cache type
    if hl_keywords or ll_keywords:
        cache_data = {
            "high_level_keywords": hl_keywords,
            "low_level_keywords": ll_keywords,
        }
        if hashing_kv.global_config.get("enable_llm_cache"):
            # Save to cache with query parameters
            queryparam_dict = {
                "mode": param.mode,
                "response_type": param.response_type,
                "top_k": param.top_k,
                "chunk_top_k": param.chunk_top_k,
                "max_entity_tokens": param.max_entity_tokens,
                "max_relation_tokens": param.max_relation_tokens,
                "max_total_tokens": param.max_total_tokens,
                "user_prompt": param.user_prompt or "",
                "enable_rerank": param.enable_rerank,
            }
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=json.dumps(cache_data),
                    prompt=text,
                    mode=param.mode,
                    cache_type="keywords",
                    queryparam=queryparam_dict,
                ),
            )

    return hl_keywords, ll_keywords


async def _get_vector_context(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    query_embedding: list[float] = None,
) -> list[dict]:
    """
    Retrieve text chunks from the vector database without reranking or truncation.

    This function performs vector search to find relevant text chunks for a query.
    Reranking and truncation will be handled later in the unified processing.

    Args:
        query: The query string to search for
        chunks_vdb: Vector database containing document chunks
        query_param: Query parameters including chunk_top_k and ids
        query_embedding: Optional pre-computed query embedding to avoid redundant embedding calls

    Returns:
        List of text chunks with metadata
    """
    try:
        # Use chunk_top_k if specified, otherwise fall back to top_k
        search_top_k = query_param.chunk_top_k or query_param.top_k
        cosine_threshold = chunks_vdb.cosine_better_than_threshold

        results = await chunks_vdb.query(
            query, top_k=search_top_k, query_embedding=query_embedding
        )
        if not results:
            logger.info(
                f"Naive query: 0 chunks (chunk_top_k:{search_top_k} cosine:{cosine_threshold})"
            )
            return []

        valid_chunks = []
        for result in results:
            if "content" in result:
                chunk_with_metadata = {
                    "content": result["content"],
                    "created_at": result.get("created_at", None),
                    "file_path": result.get("file_path", "unknown_source"),
                    "source_type": "vector",  # Mark the source type
                    "chunk_id": result.get("id"),  # Add chunk_id for deduplication
                }
                valid_chunks.append(chunk_with_metadata)

        logger.info(
            f"Naive query: {len(valid_chunks)} chunks (chunk_top_k:{search_top_k} cosine:{cosine_threshold})"
        )
        return valid_chunks

    except Exception as e:
        logger.error(f"Error in _get_vector_context: {e}")
        return []


async def _perform_kg_search(
    query: str,
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    chunks_vdb: BaseVectorStorage = None,
) -> dict[str, Any]:
    """
    Pure search logic that retrieves raw entities, relations, and vector chunks.
    No token truncation or formatting - just raw search results.
    """

    # Initialize result containers
    local_entities = []
    local_relations = []
    global_entities = []
    global_relations = []
    vector_chunks = []
    chunk_tracking = {}

    # Handle different query modes

    # Track chunk sources and metadata for final logging
    chunk_tracking = {}  # chunk_id -> {source, frequency, order}

    # Pre-compute query embedding once for all vector operations
    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    query_embedding = None
    if query and (kg_chunk_pick_method == "VECTOR" or chunks_vdb):
        actual_embedding_func = text_chunks_db.embedding_func
        if actual_embedding_func:
            try:
                query_embedding = await actual_embedding_func([query])
                query_embedding = query_embedding[
                    0
                ]  # Extract first embedding from batch result
                logger.debug("Pre-computed query embedding for all vector operations")
            except Exception as e:
                logger.warning(f"Failed to pre-compute query embedding: {e}")
                query_embedding = None

    # Handle local and global modes
    if query_param.mode == "local" and len(ll_keywords) > 0:
        local_entities, local_relations = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            query_param,
        )

    elif query_param.mode == "global" and len(hl_keywords) > 0:
        global_relations, global_entities = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            query_param,
        )

    else:  # hybrid or mix mode
        if len(ll_keywords) > 0:
            local_entities, local_relations = await _get_node_data(
                ll_keywords,
                knowledge_graph_inst,
                entities_vdb,
                query_param,
            )
        if len(hl_keywords) > 0:
            global_relations, global_entities = await _get_edge_data(
                hl_keywords,
                knowledge_graph_inst,
                relationships_vdb,
                query_param,
            )

        # Get vector chunks for mix mode
        if query_param.mode == "mix" and chunks_vdb:
            vector_chunks = await _get_vector_context(
                query,
                chunks_vdb,
                query_param,
                query_embedding,
            )
            # Track vector chunks with source metadata
            for i, chunk in enumerate(vector_chunks):
                chunk_id = chunk.get("chunk_id") or chunk.get("id")
                if chunk_id:
                    chunk_tracking[chunk_id] = {
                        "source": "C",
                        "frequency": 1,  # Vector chunks always have frequency 1
                        "order": i + 1,  # 1-based order in vector search results
                    }
                else:
                    logger.warning(f"Vector chunk missing chunk_id: {chunk}")

    # Round-robin merge entities
    final_entities = []
    seen_entities = set()
    max_len = max(len(local_entities), len(global_entities))
    for i in range(max_len):
        # First from local
        if i < len(local_entities):
            entity = local_entities[i]
            entity_name = entity.get("entity_name")
            if entity_name and entity_name not in seen_entities:
                final_entities.append(entity)
                seen_entities.add(entity_name)

        # Then from global
        if i < len(global_entities):
            entity = global_entities[i]
            entity_name = entity.get("entity_name")
            if entity_name and entity_name not in seen_entities:
                final_entities.append(entity)
                seen_entities.add(entity_name)

    # Round-robin merge relations
    final_relations = []
    seen_relations = set()
    max_len = max(len(local_relations), len(global_relations))
    for i in range(max_len):
        # First from local
        if i < len(local_relations):
            relation = local_relations[i]
            # Build relation unique identifier
            if "src_tgt" in relation:
                rel_key = tuple(sorted(relation["src_tgt"]))
            else:
                rel_key = tuple(
                    sorted([relation.get("src_id"), relation.get("tgt_id")])
                )

            if rel_key not in seen_relations:
                final_relations.append(relation)
                seen_relations.add(rel_key)

        # Then from global
        if i < len(global_relations):
            relation = global_relations[i]
            # Build relation unique identifier
            if "src_tgt" in relation:
                rel_key = tuple(sorted(relation["src_tgt"]))
            else:
                rel_key = tuple(
                    sorted([relation.get("src_id"), relation.get("tgt_id")])
                )

            if rel_key not in seen_relations:
                final_relations.append(relation)
                seen_relations.add(rel_key)

    logger.info(
        f"Raw search results: {len(final_entities)} entities, {len(final_relations)} relations, {len(vector_chunks)} vector chunks"
    )

    return {
        "final_entities": final_entities,
        "final_relations": final_relations,
        "vector_chunks": vector_chunks,
        "chunk_tracking": chunk_tracking,
        "query_embedding": query_embedding,
    }


async def _apply_token_truncation(
    search_result: dict[str, Any],
    query_param: QueryParam,
    global_config: dict[str, str],
) -> dict[str, Any]:
    """
    Apply token-based truncation to entities and relations for LLM efficiency.
    """
    tokenizer = global_config.get("tokenizer")
    if not tokenizer:
        logger.warning("No tokenizer found, skipping truncation")
        return {
            "entities_context": [],
            "relations_context": [],
            "filtered_entities": search_result["final_entities"],
            "filtered_relations": search_result["final_relations"],
            "entity_id_to_original": {},
            "relation_id_to_original": {},
        }

    # Get token limits from query_param with fallbacks
    max_entity_tokens = getattr(
        query_param,
        "max_entity_tokens",
        global_config.get("max_entity_tokens", DEFAULT_MAX_ENTITY_TOKENS),
    )
    max_relation_tokens = getattr(
        query_param,
        "max_relation_tokens",
        global_config.get("max_relation_tokens", DEFAULT_MAX_RELATION_TOKENS),
    )

    final_entities = search_result["final_entities"]
    final_relations = search_result["final_relations"]

    # Create mappings from entity/relation identifiers to original data
    entity_id_to_original = {}
    relation_id_to_original = {}

    # Generate entities context for truncation
    entities_context = []
    for i, entity in enumerate(final_entities):
        entity_name = entity["entity_name"]
        created_at = entity.get("created_at", "UNKNOWN")
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Store mapping from entity name to original data
        entity_id_to_original[entity_name] = entity

        entities_context.append(
            {
                "entity": entity_name,
                "type": entity.get("entity_type", "UNKNOWN"),
                "description": entity.get("description", "UNKNOWN"),
                "created_at": created_at,
                "file_path": entity.get("file_path", "unknown_source"),
            }
        )

    # Generate relations context for truncation
    relations_context = []
    for i, relation in enumerate(final_relations):
        created_at = relation.get("created_at", "UNKNOWN")
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Handle different relation data formats
        if "src_tgt" in relation:
            entity1, entity2 = relation["src_tgt"]
        else:
            entity1, entity2 = relation.get("src_id"), relation.get("tgt_id")

        # Store mapping from relation pair to original data
        relation_key = (entity1, entity2)
        relation_id_to_original[relation_key] = relation

        relations_context.append(
            {
                "entity1": entity1,
                "entity2": entity2,
                "description": relation.get("description", "UNKNOWN"),
                "created_at": created_at,
                "file_path": relation.get("file_path", "unknown_source"),
            }
        )

    logger.debug(
        f"Before truncation: {len(entities_context)} entities, {len(relations_context)} relations"
    )

    # Apply token-based truncation
    if entities_context:
        # Remove file_path and created_at for token calculation
        entities_context_for_truncation = []
        for entity in entities_context:
            entity_copy = entity.copy()
            entity_copy.pop("file_path", None)
            entity_copy.pop("created_at", None)
            entities_context_for_truncation.append(entity_copy)

        entities_context = truncate_list_by_token_size(
            entities_context_for_truncation,
            key=lambda x: "\n".join(
                json.dumps(item, ensure_ascii=False) for item in [x]
            ),
            max_token_size=max_entity_tokens,
            tokenizer=tokenizer,
        )

    if relations_context:
        # Remove file_path and created_at for token calculation
        relations_context_for_truncation = []
        for relation in relations_context:
            relation_copy = relation.copy()
            relation_copy.pop("file_path", None)
            relation_copy.pop("created_at", None)
            relations_context_for_truncation.append(relation_copy)

        relations_context = truncate_list_by_token_size(
            relations_context_for_truncation,
            key=lambda x: "\n".join(
                json.dumps(item, ensure_ascii=False) for item in [x]
            ),
            max_token_size=max_relation_tokens,
            tokenizer=tokenizer,
        )

    logger.info(
        f"After truncation: {len(entities_context)} entities, {len(relations_context)} relations"
    )

    # Create filtered original data based on truncated context
    filtered_entities = []
    filtered_entity_id_to_original = {}
    if entities_context:
        final_entity_names = {e["entity"] for e in entities_context}
        seen_nodes = set()
        for entity in final_entities:
            name = entity.get("entity_name")
            if name in final_entity_names and name not in seen_nodes:
                filtered_entities.append(entity)
                filtered_entity_id_to_original[name] = entity
                seen_nodes.add(name)

    filtered_relations = []
    filtered_relation_id_to_original = {}
    if relations_context:
        final_relation_pairs = {(r["entity1"], r["entity2"]) for r in relations_context}
        seen_edges = set()
        for relation in final_relations:
            src, tgt = relation.get("src_id"), relation.get("tgt_id")
            if src is None or tgt is None:
                src, tgt = relation.get("src_tgt", (None, None))

            pair = (src, tgt)
            if pair in final_relation_pairs and pair not in seen_edges:
                filtered_relations.append(relation)
                filtered_relation_id_to_original[pair] = relation
                seen_edges.add(pair)

    return {
        "entities_context": entities_context,
        "relations_context": relations_context,
        "filtered_entities": filtered_entities,
        "filtered_relations": filtered_relations,
        "entity_id_to_original": filtered_entity_id_to_original,
        "relation_id_to_original": filtered_relation_id_to_original,
    }


async def _merge_all_chunks(
    filtered_entities: list[dict],
    filtered_relations: list[dict],
    vector_chunks: list[dict],
    query: str = "",
    knowledge_graph_inst: BaseGraphStorage = None,
    text_chunks_db: BaseKVStorage = None,
    query_param: QueryParam = None,
    chunks_vdb: BaseVectorStorage = None,
    chunk_tracking: dict = None,
    query_embedding: list[float] = None,
) -> list[dict]:
    """
    Merge chunks from different sources: vector_chunks + entity_chunks + relation_chunks.
    """
    if chunk_tracking is None:
        chunk_tracking = {}

    # Get chunks from entities
    entity_chunks = []
    if filtered_entities and text_chunks_db:
        entity_chunks = await _find_related_text_unit_from_entities(
            filtered_entities,
            query_param,
            text_chunks_db,
            knowledge_graph_inst,
            query,
            chunks_vdb,
            chunk_tracking=chunk_tracking,
            query_embedding=query_embedding,
        )

    # Get chunks from relations
    relation_chunks = []
    if filtered_relations and text_chunks_db:
        relation_chunks = await _find_related_text_unit_from_relations(
            filtered_relations,
            query_param,
            text_chunks_db,
            entity_chunks,  # For deduplication
            query,
            chunks_vdb,
            chunk_tracking=chunk_tracking,
            query_embedding=query_embedding,
        )

    # Round-robin merge chunks from different sources with deduplication
    merged_chunks = []
    seen_chunk_ids = set()
    max_len = max(len(vector_chunks), len(entity_chunks), len(relation_chunks))
    origin_len = len(vector_chunks) + len(entity_chunks) + len(relation_chunks)

    for i in range(max_len):
        # Add from vector chunks first (Naive mode)
        if i < len(vector_chunks):
            chunk = vector_chunks[i]
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(
                    {
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                    }
                )

        # Add from entity chunks (Local mode)
        if i < len(entity_chunks):
            chunk = entity_chunks[i]
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(
                    {
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                    }
                )

        # Add from relation chunks (Global mode)
        if i < len(relation_chunks):
            chunk = relation_chunks[i]
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(
                    {
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                    }
                )

    logger.info(
        f"Round-robin merged chunks: {origin_len} -> {len(merged_chunks)} (deduplicated {origin_len - len(merged_chunks)})"
    )

    return merged_chunks


async def _build_context_str(
    entities_context: list[dict],
    relations_context: list[dict],
    merged_chunks: list[dict],
    query: str,
    query_param: QueryParam,
    global_config: dict[str, str],
    chunk_tracking: dict = None,
    entity_id_to_original: dict = None,
    relation_id_to_original: dict = None,
) -> tuple[str, dict[str, Any]]:
    """
    Build the final LLM context string with token processing.
    This includes dynamic token calculation and final chunk truncation.
    """
    tokenizer = global_config.get("tokenizer")
    if not tokenizer:
        logger.error("Missing tokenizer, cannot build LLM context")
        # Return empty raw data structure when no tokenizer
        empty_raw_data = convert_to_user_format(
            [],
            [],
            [],
            [],
            query_param.mode,
        )
        empty_raw_data["status"] = "failure"
        empty_raw_data["message"] = "Missing tokenizer, cannot build LLM context."
        return "", empty_raw_data

    # Get token limits
    max_total_tokens = getattr(
        query_param,
        "max_total_tokens",
        global_config.get("max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS),
    )

    # Get the system prompt template from PROMPTS or global_config
    sys_prompt_template = global_config.get(
        "system_prompt_template", PROMPTS["rag_response"]
    )

    kg_context_template = PROMPTS["kg_query_context"]
    user_prompt = query_param.user_prompt if query_param.user_prompt else ""
    response_type = (
        query_param.response_type
        if query_param.response_type
        else "Multiple Paragraphs"
    )

    entities_str = "\n".join(
        json.dumps(entity, ensure_ascii=False) for entity in entities_context
    )
    relations_str = "\n".join(
        json.dumps(relation, ensure_ascii=False) for relation in relations_context
    )

    # Calculate preliminary kg context tokens
    pre_kg_context = kg_context_template.format(
        entities_str=entities_str,
        relations_str=relations_str,
        text_chunks_str="",
        reference_list_str="",
    )
    kg_context_tokens = len(tokenizer.encode(pre_kg_context))

    # Calculate preliminary system prompt tokens
    pre_sys_prompt = sys_prompt_template.format(
        context_data="",  # Empty for overhead calculation
        response_type=response_type,
        user_prompt=user_prompt,
    )
    sys_prompt_tokens = len(tokenizer.encode(pre_sys_prompt))

    # Calculate available tokens for text chunks
    query_tokens = len(tokenizer.encode(query))
    buffer_tokens = 200  # reserved for reference list and safety buffer
    available_chunk_tokens = max_total_tokens - (
        sys_prompt_tokens + kg_context_tokens + query_tokens + buffer_tokens
    )

    logger.debug(
        f"Token allocation - Total: {max_total_tokens}, SysPrompt: {sys_prompt_tokens}, Query: {query_tokens}, KG: {kg_context_tokens}, Buffer: {buffer_tokens}, Available for chunks: {available_chunk_tokens}"
    )

    # Apply token truncation to chunks using the dynamic limit
    truncated_chunks = await process_chunks_unified(
        query=query,
        unique_chunks=merged_chunks,
        query_param=query_param,
        global_config=global_config,
        source_type=query_param.mode,
        chunk_token_limit=available_chunk_tokens,  # Pass dynamic limit
    )

    # Generate reference list from truncated chunks using the new common function
    reference_list, truncated_chunks = generate_reference_list_from_chunks(
        truncated_chunks
    )

    # Rebuild chunks_context with truncated chunks
    # The actual tokens may be slightly less than available_chunk_tokens due to deduplication logic
    chunks_context = []
    for i, chunk in enumerate(truncated_chunks):
        chunks_context.append(
            {
                "reference_id": chunk["reference_id"],
                "content": chunk["content"],
            }
        )

    text_units_str = "\n".join(
        json.dumps(text_unit, ensure_ascii=False) for text_unit in chunks_context
    )
    reference_list_str = "\n".join(
        f"[{ref['reference_id']}] {ref['file_path']}"
        for ref in reference_list
        if ref["reference_id"]
    )

    logger.info(
        f"Final context: {len(entities_context)} entities, {len(relations_context)} relations, {len(chunks_context)} chunks"
    )

    # not necessary to use LLM to generate a response
    if not entities_context and not relations_context and not chunks_context:
        # Return empty raw data structure when no entities/relations
        empty_raw_data = convert_to_user_format(
            [],
            [],
            [],
            [],
            query_param.mode,
        )
        empty_raw_data["status"] = "failure"
        empty_raw_data["message"] = "Query returned empty dataset."
        return "", empty_raw_data

    # output chunks tracking infomations
    # format: <source><frequency>/<order> (e.g., E5/2 R2/1 C1/1)
    if truncated_chunks and chunk_tracking:
        chunk_tracking_log = []
        for chunk in truncated_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id and chunk_id in chunk_tracking:
                tracking_info = chunk_tracking[chunk_id]
                source = tracking_info["source"]
                frequency = tracking_info["frequency"]
                order = tracking_info["order"]
                chunk_tracking_log.append(f"{source}{frequency}/{order}")
            else:
                chunk_tracking_log.append("?0/0")

        if chunk_tracking_log:
            logger.info(f"Final chunks S+F/O: {' '.join(chunk_tracking_log)}")

    result = kg_context_template.format(
        entities_str=entities_str,
        relations_str=relations_str,
        text_chunks_str=text_units_str,
        reference_list_str=reference_list_str,
    )

    # Always return both context and complete data structure (unified approach)
    logger.debug(
        f"[_build_context_str] Converting to user format: {len(entities_context)} entities, {len(relations_context)} relations, {len(truncated_chunks)} chunks"
    )
    final_data = convert_to_user_format(
        entities_context,
        relations_context,
        truncated_chunks,
        reference_list,
        query_param.mode,
        entity_id_to_original,
        relation_id_to_original,
    )
    logger.debug(
        f"[_build_context_str] Final data after conversion: {len(final_data.get('entities', []))} entities, {len(final_data.get('relationships', []))} relationships, {len(final_data.get('chunks', []))} chunks"
    )
    return result, final_data


# Now let's update the old _build_query_context to use the new architecture
async def _build_query_context(
    query: str,
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    chunks_vdb: BaseVectorStorage = None,
) -> QueryContextResult | None:
    """
    Main query context building function using the new 4-stage architecture:
    1. Search -> 2. Truncate -> 3. Merge chunks -> 4. Build LLM context

    Returns unified QueryContextResult containing both context and raw_data.
    """

    if not query:
        logger.warning("Query is empty, skipping context building")
        return None

    # Stage 1: Pure search
    search_result = await _perform_kg_search(
        query,
        ll_keywords,
        hl_keywords,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        chunks_vdb,
    )

    if not search_result["final_entities"] and not search_result["final_relations"]:
        if query_param.mode != "mix":
            return None
        else:
            if not search_result["chunk_tracking"]:
                return None

    # Stage 2: Apply token truncation for LLM efficiency
    truncation_result = await _apply_token_truncation(
        search_result,
        query_param,
        text_chunks_db.global_config,
    )

    # Stage 3: Merge chunks using filtered entities/relations
    merged_chunks = await _merge_all_chunks(
        filtered_entities=truncation_result["filtered_entities"],
        filtered_relations=truncation_result["filtered_relations"],
        vector_chunks=search_result["vector_chunks"],
        query=query,
        knowledge_graph_inst=knowledge_graph_inst,
        text_chunks_db=text_chunks_db,
        query_param=query_param,
        chunks_vdb=chunks_vdb,
        chunk_tracking=search_result["chunk_tracking"],
        query_embedding=search_result["query_embedding"],
    )

    if (
        not merged_chunks
        and not truncation_result["entities_context"]
        and not truncation_result["relations_context"]
    ):
        return None

    # Stage 4: Build final LLM context with dynamic token processing
    # _build_context_str now always returns tuple[str, dict]
    context, raw_data = await _build_context_str(
        entities_context=truncation_result["entities_context"],
        relations_context=truncation_result["relations_context"],
        merged_chunks=merged_chunks,
        query=query,
        query_param=query_param,
        global_config=text_chunks_db.global_config,
        chunk_tracking=search_result["chunk_tracking"],
        entity_id_to_original=truncation_result["entity_id_to_original"],
        relation_id_to_original=truncation_result["relation_id_to_original"],
    )

    # Convert keywords strings to lists and add complete metadata to raw_data
    hl_keywords_list = hl_keywords.split(", ") if hl_keywords else []
    ll_keywords_list = ll_keywords.split(", ") if ll_keywords else []

    # Add complete metadata to raw_data (preserve existing metadata including query_mode)
    if "metadata" not in raw_data:
        raw_data["metadata"] = {}

    # Update keywords while preserving existing metadata
    raw_data["metadata"]["keywords"] = {
        "high_level": hl_keywords_list,
        "low_level": ll_keywords_list,
    }
    raw_data["metadata"]["processing_info"] = {
        "total_entities_found": len(search_result.get("final_entities", [])),
        "total_relations_found": len(search_result.get("final_relations", [])),
        "entities_after_truncation": len(
            truncation_result.get("filtered_entities", [])
        ),
        "relations_after_truncation": len(
            truncation_result.get("filtered_relations", [])
        ),
        "merged_chunks_count": len(merged_chunks),
        "final_chunks_count": len(raw_data.get("data", {}).get("chunks", [])),
    }

    logger.debug(
        f"[_build_query_context] Context length: {len(context) if context else 0}"
    )
    logger.debug(
        f"[_build_query_context] Raw data entities: {len(raw_data.get('data', {}).get('entities', []))}, relationships: {len(raw_data.get('data', {}).get('relationships', []))}, chunks: {len(raw_data.get('data', {}).get('chunks', []))}"
    )

    return QueryContextResult(context=context, raw_data=raw_data)


async def _get_node_data(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    query_param: QueryParam,
):
    # get similar entities
    logger.info(
        f"Query nodes: {query} (top_k:{query_param.top_k}, cosine:{entities_vdb.cosine_better_than_threshold})"
    )

    results = await entities_vdb.query(query, top_k=query_param.top_k)

    if not len(results):
        return [], []

    # Extract all entity IDs from your results list
    node_ids = [r["entity_name"] for r in results]

    # Call the batch node retrieval and degree functions concurrently.
    nodes_dict, degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_nodes_batch(node_ids),
        knowledge_graph_inst.node_degrees_batch(node_ids),
    )

    # Now, if you need the node data and degree in order:
    node_datas = [nodes_dict.get(nid) for nid in node_ids]
    node_degrees = [degrees_dict.get(nid, 0) for nid in node_ids]

    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_datas = [
        {
            **n,
            "entity_name": k["entity_name"],
            "rank": d,
            "created_at": k.get("created_at"),
        }
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]

    use_relations = await _find_most_related_edges_from_entities(
        node_datas,
        query_param,
        knowledge_graph_inst,
    )

    logger.info(
        f"Local query: {len(node_datas)} entites, {len(use_relations)} relations"
    )

    # Entities are sorted by cosine similarity
    # Relations are sorted by rank + weight
    return node_datas, use_relations


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    node_names = [dp["entity_name"] for dp in node_datas]
    batch_edges_dict = await knowledge_graph_inst.get_nodes_edges_batch(node_names)

    all_edges = []
    seen = set()

    for node_name in node_names:
        this_edges = batch_edges_dict.get(node_name, [])
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    # Prepare edge pairs in two forms:
    # For the batch edge properties function, use dicts.
    edge_pairs_dicts = [{"src": e[0], "tgt": e[1]} for e in all_edges]
    # For edge degrees, use tuples.
    edge_pairs_tuples = list(all_edges)  # all_edges is already a list of tuples

    # Call the batched functions concurrently.
    edge_data_dict, edge_degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_edges_batch(edge_pairs_dicts),
        knowledge_graph_inst.edge_degrees_batch(edge_pairs_tuples),
    )

    # Reconstruct edge_datas list in the same order as the deduplicated results.
    all_edges_data = []
    for pair in all_edges:
        edge_props = edge_data_dict.get(pair)
        if edge_props is not None:
            if "weight" not in edge_props:
                logger.warning(
                    f"Edge {pair} missing 'weight' attribute, using default value 1.0"
                )
                edge_props["weight"] = 1.0

            combined = {
                "src_tgt": pair,
                "rank": edge_degrees_dict.get(pair, 0),
                **edge_props,
            }
            all_edges_data.append(combined)

    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )

    return all_edges_data


async def _find_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
    query: str = None,
    chunks_vdb: BaseVectorStorage = None,
    chunk_tracking: dict = None,
    query_embedding=None,
):
    """
    Find text chunks related to entities using configurable chunk selection method.

    This function supports two chunk selection strategies:
    1. WEIGHT: Linear gradient weighted polling based on chunk occurrence count
    2. VECTOR: Vector similarity-based selection using embedding cosine similarity
    """
    logger.debug(f"Finding text chunks from {len(node_datas)} entities")

    if not node_datas:
        return []

    # Step 1: Collect all text chunks for each entity
    entities_with_chunks = []
    for entity in node_datas:
        if entity.get("source_id"):
            chunks = split_string_by_multi_markers(
                entity["source_id"], [GRAPH_FIELD_SEP]
            )
            if chunks:
                entities_with_chunks.append(
                    {
                        "entity_name": entity["entity_name"],
                        "chunks": chunks,
                        "entity_data": entity,
                    }
                )

    if not entities_with_chunks:
        logger.warning("No entities with text chunks found")
        return []

    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    max_related_chunks = text_chunks_db.global_config.get(
        "related_chunk_number", DEFAULT_RELATED_CHUNK_NUMBER
    )

    # Step 2: Count chunk occurrences and deduplicate (keep chunks from earlier positioned entities)
    chunk_occurrence_count = {}
    for entity_info in entities_with_chunks:
        deduplicated_chunks = []
        for chunk_id in entity_info["chunks"]:
            chunk_occurrence_count[chunk_id] = (
                chunk_occurrence_count.get(chunk_id, 0) + 1
            )

            # If this is the first occurrence (count == 1), keep it; otherwise skip (duplicate from later position)
            if chunk_occurrence_count[chunk_id] == 1:
                deduplicated_chunks.append(chunk_id)
            # count > 1 means this chunk appeared in an earlier entity, so skip it

        # Update entity's chunks to deduplicated chunks
        entity_info["chunks"] = deduplicated_chunks

    # Step 3: Sort chunks for each entity by occurrence count (higher count = higher priority)
    total_entity_chunks = 0
    for entity_info in entities_with_chunks:
        sorted_chunks = sorted(
            entity_info["chunks"],
            key=lambda chunk_id: chunk_occurrence_count.get(chunk_id, 0),
            reverse=True,
        )
        entity_info["sorted_chunks"] = sorted_chunks
        total_entity_chunks += len(sorted_chunks)

    selected_chunk_ids = []  # Initialize to avoid UnboundLocalError

    # Step 4: Apply the selected chunk selection algorithm
    # Pick by vector similarity:
    #     The order of text chunks aligns with the naive retrieval's destination.
    #     When reranking is disabled, the text chunks delivered to the LLM tend to favor naive retrieval.
    if kg_chunk_pick_method == "VECTOR" and query and chunks_vdb:
        num_of_chunks = int(max_related_chunks * len(entities_with_chunks) / 2)

        # Get embedding function from global config
        actual_embedding_func = text_chunks_db.embedding_func
        if not actual_embedding_func:
            logger.warning("No embedding function found, falling back to WEIGHT method")
            kg_chunk_pick_method = "WEIGHT"
        else:
            try:
                selected_chunk_ids = await pick_by_vector_similarity(
                    query=query,
                    text_chunks_storage=text_chunks_db,
                    chunks_vdb=chunks_vdb,
                    num_of_chunks=num_of_chunks,
                    entity_info=entities_with_chunks,
                    embedding_func=actual_embedding_func,
                    query_embedding=query_embedding,
                )

                if selected_chunk_ids == []:
                    kg_chunk_pick_method = "WEIGHT"
                    logger.warning(
                        "No entity-related chunks selected by vector similarity, falling back to WEIGHT method"
                    )
                else:
                    logger.info(
                        f"Selecting {len(selected_chunk_ids)} from {total_entity_chunks} entity-related chunks by vector similarity"
                    )

            except Exception as e:
                logger.error(
                    f"Error in vector similarity sorting: {e}, falling back to WEIGHT method"
                )
                kg_chunk_pick_method = "WEIGHT"

    if kg_chunk_pick_method == "WEIGHT":
        # Pick by entity and chunk weight:
        #     When reranking is disabled, delivered more solely KG related chunks to the LLM
        selected_chunk_ids = pick_by_weighted_polling(
            entities_with_chunks, max_related_chunks, min_related_chunks=1
        )

        logger.info(
            f"Selecting {len(selected_chunk_ids)} from {total_entity_chunks} entity-related chunks by weighted polling"
        )

    if not selected_chunk_ids:
        return []

    # Step 5: Batch retrieve chunk data
    unique_chunk_ids = list(
        dict.fromkeys(selected_chunk_ids)
    )  # Remove duplicates while preserving order
    chunk_data_list = await text_chunks_db.get_by_ids(unique_chunk_ids)

    # Step 6: Build result chunks with valid data and update chunk tracking
    result_chunks = []
    for i, (chunk_id, chunk_data) in enumerate(zip(unique_chunk_ids, chunk_data_list)):
        if chunk_data is not None and "content" in chunk_data:
            chunk_data_copy = chunk_data.copy()
            chunk_data_copy["source_type"] = "entity"
            chunk_data_copy["chunk_id"] = chunk_id  # Add chunk_id for deduplication
            result_chunks.append(chunk_data_copy)

            # Update chunk tracking if provided
            if chunk_tracking is not None:
                chunk_tracking[chunk_id] = {
                    "source": "E",
                    "frequency": chunk_occurrence_count.get(chunk_id, 1),
                    "order": i + 1,  # 1-based order in final entity-related results
                }

    return result_chunks


async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    query_param: QueryParam,
):
    logger.info(
        f"Query edges: {keywords} (top_k:{query_param.top_k}, cosine:{relationships_vdb.cosine_better_than_threshold})"
    )

    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)

    if not len(results):
        return [], []

    # Prepare edge pairs in two forms:
    # For the batch edge properties function, use dicts.
    edge_pairs_dicts = [{"src": r["src_id"], "tgt": r["tgt_id"]} for r in results]
    edge_data_dict = await knowledge_graph_inst.get_edges_batch(edge_pairs_dicts)

    # Reconstruct edge_datas list in the same order as results.
    edge_datas = []
    for k in results:
        pair = (k["src_id"], k["tgt_id"])
        edge_props = edge_data_dict.get(pair)
        if edge_props is not None:
            if "weight" not in edge_props:
                logger.warning(
                    f"Edge {pair} missing 'weight' attribute, using default value 1.0"
                )
                edge_props["weight"] = 1.0

            # Keep edge data without rank, maintain vector search order
            combined = {
                "src_id": k["src_id"],
                "tgt_id": k["tgt_id"],
                "created_at": k.get("created_at", None),
                **edge_props,
            }
            edge_datas.append(combined)

    # Relations maintain vector search order (sorted by similarity)

    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas,
        query_param,
        knowledge_graph_inst,
    )

    logger.info(
        f"Global query: {len(use_entities)} entites, {len(edge_datas)} relations"
    )

    return edge_datas, use_entities


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    # Only get nodes data, no need for node degrees
    nodes_dict = await knowledge_graph_inst.get_nodes_batch(entity_names)

    # Rebuild the list in the same order as entity_names
    node_datas = []
    for entity_name in entity_names:
        node = nodes_dict.get(entity_name)
        if node is None:
            logger.warning(f"Node '{entity_name}' not found in batch retrieval.")
            continue
        # Combine the node data with the entity name, no rank needed
        combined = {**node, "entity_name": entity_name}
        node_datas.append(combined)

    return node_datas


async def _find_related_text_unit_from_relations(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    entity_chunks: list[dict] = None,
    query: str = None,
    chunks_vdb: BaseVectorStorage = None,
    chunk_tracking: dict = None,
    query_embedding=None,
):
    """
    Find text chunks related to relationships using configurable chunk selection method.

    This function supports two chunk selection strategies:
    1. WEIGHT: Linear gradient weighted polling based on chunk occurrence count
    2. VECTOR: Vector similarity-based selection using embedding cosine similarity
    """
    logger.debug(f"Finding text chunks from {len(edge_datas)} relations")

    if not edge_datas:
        return []

    # Step 1: Collect all text chunks for each relationship
    relations_with_chunks = []
    for relation in edge_datas:
        if relation.get("source_id"):
            chunks = split_string_by_multi_markers(
                relation["source_id"], [GRAPH_FIELD_SEP]
            )
            if chunks:
                # Build relation identifier
                if "src_tgt" in relation:
                    rel_key = tuple(sorted(relation["src_tgt"]))
                else:
                    rel_key = tuple(
                        sorted([relation.get("src_id"), relation.get("tgt_id")])
                    )

                relations_with_chunks.append(
                    {
                        "relation_key": rel_key,
                        "chunks": chunks,
                        "relation_data": relation,
                    }
                )

    if not relations_with_chunks:
        logger.warning("No relation-related chunks found")
        return []

    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    max_related_chunks = text_chunks_db.global_config.get(
        "related_chunk_number", DEFAULT_RELATED_CHUNK_NUMBER
    )

    # Step 2: Count chunk occurrences and deduplicate (keep chunks from earlier positioned relationships)
    # Also remove duplicates with entity_chunks

    # Extract chunk IDs from entity_chunks for deduplication
    entity_chunk_ids = set()
    if entity_chunks:
        for chunk in entity_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                entity_chunk_ids.add(chunk_id)

    chunk_occurrence_count = {}
    # Track unique chunk_ids that have been removed to avoid double counting
    removed_entity_chunk_ids = set()

    for relation_info in relations_with_chunks:
        deduplicated_chunks = []
        for chunk_id in relation_info["chunks"]:
            # Skip chunks that already exist in entity_chunks
            if chunk_id in entity_chunk_ids:
                # Only count each unique chunk_id once
                removed_entity_chunk_ids.add(chunk_id)
                continue

            chunk_occurrence_count[chunk_id] = (
                chunk_occurrence_count.get(chunk_id, 0) + 1
            )

            # If this is the first occurrence (count == 1), keep it; otherwise skip (duplicate from later position)
            if chunk_occurrence_count[chunk_id] == 1:
                deduplicated_chunks.append(chunk_id)
            # count > 1 means this chunk appeared in an earlier relationship, so skip it

        # Update relationship's chunks to deduplicated chunks
        relation_info["chunks"] = deduplicated_chunks

    # Check if any relations still have chunks after deduplication
    relations_with_chunks = [
        relation_info
        for relation_info in relations_with_chunks
        if relation_info["chunks"]
    ]

    if not relations_with_chunks:
        logger.info(
            f"Find no additional relations-related chunks from {len(edge_datas)} relations"
        )
        return []

    # Step 3: Sort chunks for each relationship by occurrence count (higher count = higher priority)
    total_relation_chunks = 0
    for relation_info in relations_with_chunks:
        sorted_chunks = sorted(
            relation_info["chunks"],
            key=lambda chunk_id: chunk_occurrence_count.get(chunk_id, 0),
            reverse=True,
        )
        relation_info["sorted_chunks"] = sorted_chunks
        total_relation_chunks += len(sorted_chunks)

    logger.info(
        f"Find {total_relation_chunks} additional chunks in {len(relations_with_chunks)} relations (deduplicated {len(removed_entity_chunk_ids)})"
    )

    # Step 4: Apply the selected chunk selection algorithm
    selected_chunk_ids = []  # Initialize to avoid UnboundLocalError

    if kg_chunk_pick_method == "VECTOR" and query and chunks_vdb:
        num_of_chunks = int(max_related_chunks * len(relations_with_chunks) / 2)

        # Get embedding function from global config
        actual_embedding_func = text_chunks_db.embedding_func
        if not actual_embedding_func:
            logger.warning("No embedding function found, falling back to WEIGHT method")
            kg_chunk_pick_method = "WEIGHT"
        else:
            try:
                selected_chunk_ids = await pick_by_vector_similarity(
                    query=query,
                    text_chunks_storage=text_chunks_db,
                    chunks_vdb=chunks_vdb,
                    num_of_chunks=num_of_chunks,
                    entity_info=relations_with_chunks,
                    embedding_func=actual_embedding_func,
                    query_embedding=query_embedding,
                )

                if selected_chunk_ids == []:
                    kg_chunk_pick_method = "WEIGHT"
                    logger.warning(
                        "No relation-related chunks selected by vector similarity, falling back to WEIGHT method"
                    )
                else:
                    logger.info(
                        f"Selecting {len(selected_chunk_ids)} from {total_relation_chunks} relation-related chunks by vector similarity"
                    )

            except Exception as e:
                logger.error(
                    f"Error in vector similarity sorting: {e}, falling back to WEIGHT method"
                )
                kg_chunk_pick_method = "WEIGHT"

    if kg_chunk_pick_method == "WEIGHT":
        # Apply linear gradient weighted polling algorithm
        selected_chunk_ids = pick_by_weighted_polling(
            relations_with_chunks, max_related_chunks, min_related_chunks=1
        )

        logger.info(
            f"Selecting {len(selected_chunk_ids)} from {total_relation_chunks} relation-related chunks by weighted polling"
        )

    logger.debug(
        f"KG related chunks: {len(entity_chunks)} from entitys, {len(selected_chunk_ids)} from relations"
    )

    if not selected_chunk_ids:
        return []

    # Step 5: Batch retrieve chunk data
    unique_chunk_ids = list(
        dict.fromkeys(selected_chunk_ids)
    )  # Remove duplicates while preserving order
    chunk_data_list = await text_chunks_db.get_by_ids(unique_chunk_ids)

    # Step 6: Build result chunks with valid data and update chunk tracking
    result_chunks = []
    for i, (chunk_id, chunk_data) in enumerate(zip(unique_chunk_ids, chunk_data_list)):
        if chunk_data is not None and "content" in chunk_data:
            chunk_data_copy = chunk_data.copy()
            chunk_data_copy["source_type"] = "relationship"
            chunk_data_copy["chunk_id"] = chunk_id  # Add chunk_id for deduplication
            result_chunks.append(chunk_data_copy)

            # Update chunk tracking if provided
            if chunk_tracking is not None:
                chunk_tracking[chunk_id] = {
                    "source": "R",
                    "frequency": chunk_occurrence_count.get(chunk_id, 1),
                    "order": i + 1,  # 1-based order in final relation-related results
                }

    return result_chunks


@overload
async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
    return_raw_data: Literal[True] = True,
) -> dict[str, Any]: ...


@overload
async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
    return_raw_data: Literal[False] = False,
) -> str | AsyncIterator[str]: ...


async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
) -> QueryResult | None:
    """
    Execute naive query and return unified QueryResult object.

    Args:
        query: Query string
        chunks_vdb: Document chunks vector database
        query_param: Query parameters
        global_config: Global configuration
        hashing_kv: Cache storage
        system_prompt: System prompt

    Returns:
        QueryResult | None: Unified query result object containing:
            - content: Non-streaming response text content
            - response_iterator: Streaming response iterator
            - raw_data: Complete structured data (including references and metadata)
            - is_streaming: Whether this is a streaming result

        Returns None when no relevant chunks are retrieved.
    """

    if not query:
        return QueryResult(content=PROMPTS["fail_response"])

    if query_param.model_func:
        use_model_func = query_param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    tokenizer: Tokenizer = global_config["tokenizer"]
    if not tokenizer:
        logger.error("Tokenizer not found in global configuration.")
        return QueryResult(content=PROMPTS["fail_response"])

    chunks = await _get_vector_context(query, chunks_vdb, query_param, None)

    if chunks is None or len(chunks) == 0:
        logger.info(
            "[naive_query] No relevant document chunks found; returning no-result."
        )
        return None

    # Calculate dynamic token limit for chunks
    max_total_tokens = getattr(
        query_param,
        "max_total_tokens",
        global_config.get("max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS),
    )

    # Calculate system prompt template tokens (excluding content_data)
    user_prompt = f"\n\n{query_param.user_prompt}" if query_param.user_prompt else "n/a"
    response_type = (
        query_param.response_type
        if query_param.response_type
        else "Multiple Paragraphs"
    )

    # Use the provided system prompt or default
    sys_prompt_template = (
        system_prompt if system_prompt else PROMPTS["naive_rag_response"]
    )

    # Create a preliminary system prompt with empty content_data to calculate overhead
    pre_sys_prompt = sys_prompt_template.format(
        response_type=response_type,
        user_prompt=user_prompt,
        content_data="",  # Empty for overhead calculation
    )

    # Calculate available tokens for chunks
    sys_prompt_tokens = len(tokenizer.encode(pre_sys_prompt))
    query_tokens = len(tokenizer.encode(query))
    buffer_tokens = 200  # reserved for reference list and safety buffer
    available_chunk_tokens = max_total_tokens - (
        sys_prompt_tokens + query_tokens + buffer_tokens
    )

    logger.debug(
        f"Naive query token allocation - Total: {max_total_tokens}, SysPrompt: {sys_prompt_tokens}, Query: {query_tokens}, Buffer: {buffer_tokens}, Available for chunks: {available_chunk_tokens}"
    )

    # Process chunks using unified processing with dynamic token limit
    processed_chunks = await process_chunks_unified(
        query=query,
        unique_chunks=chunks,
        query_param=query_param,
        global_config=global_config,
        source_type="vector",
        chunk_token_limit=available_chunk_tokens,  # Pass dynamic limit
    )

    # Generate reference list from processed chunks using the new common function
    reference_list, processed_chunks_with_ref_ids = generate_reference_list_from_chunks(
        processed_chunks
    )

    logger.info(f"Final context: {len(processed_chunks_with_ref_ids)} chunks")

    # Build raw data structure for naive mode using processed chunks with reference IDs
    raw_data = convert_to_user_format(
        [],  # naive mode has no entities
        [],  # naive mode has no relationships
        processed_chunks_with_ref_ids,
        reference_list,
        "naive",
    )

    # Add complete metadata for naive mode
    if "metadata" not in raw_data:
        raw_data["metadata"] = {}
    raw_data["metadata"]["keywords"] = {
        "high_level": [],  # naive mode has no keyword extraction
        "low_level": [],  # naive mode has no keyword extraction
    }
    raw_data["metadata"]["processing_info"] = {
        "total_chunks_found": len(chunks),
        "final_chunks_count": len(processed_chunks_with_ref_ids),
    }

    # Build chunks_context from processed chunks with reference IDs
    chunks_context = []
    for i, chunk in enumerate(processed_chunks_with_ref_ids):
        chunks_context.append(
            {
                "reference_id": chunk["reference_id"],
                "content": chunk["content"],
            }
        )

    text_units_str = "\n".join(
        json.dumps(text_unit, ensure_ascii=False) for text_unit in chunks_context
    )
    reference_list_str = "\n".join(
        f"[{ref['reference_id']}] {ref['file_path']}"
        for ref in reference_list
        if ref["reference_id"]
    )

    naive_context_template = PROMPTS["naive_query_context"]
    context_content = naive_context_template.format(
        text_chunks_str=text_units_str,
        reference_list_str=reference_list_str,
    )

    if query_param.only_need_context and not query_param.only_need_prompt:
        return QueryResult(content=context_content, raw_data=raw_data)

    sys_prompt = sys_prompt_template.format(
        response_type=query_param.response_type,
        user_prompt=user_prompt,
        content_data=context_content,
    )

    user_query = query

    if query_param.only_need_prompt:
        prompt_content = "\n\n".join([sys_prompt, "---User Query---", user_query])
        return QueryResult(content=prompt_content, raw_data=raw_data)

    # Handle cache
    args_hash = compute_args_hash(
        query_param.mode,
        query,
        query_param.response_type,
        query_param.top_k,
        query_param.chunk_top_k,
        query_param.max_entity_tokens,
        query_param.max_relation_tokens,
        query_param.max_total_tokens,
        query_param.user_prompt or "",
        query_param.enable_rerank,
    )
    cached_result = await handle_cache(
        hashing_kv, args_hash, user_query, query_param.mode, cache_type="query"
    )
    if cached_result is not None:
        cached_response, _ = cached_result  # Extract content, ignore timestamp
        logger.info(
            " == LLM cache == Query cache hit, using cached response as query result"
        )
        response = cached_response
    else:
        response = await use_model_func(
            user_query,
            system_prompt=sys_prompt,
            history_messages=query_param.conversation_history,
            enable_cot=True,
            stream=query_param.stream,
        )

        if hashing_kv and hashing_kv.global_config.get("enable_llm_cache"):
            queryparam_dict = {
                "mode": query_param.mode,
                "response_type": query_param.response_type,
                "top_k": query_param.top_k,
                "chunk_top_k": query_param.chunk_top_k,
                "max_entity_tokens": query_param.max_entity_tokens,
                "max_relation_tokens": query_param.max_relation_tokens,
                "max_total_tokens": query_param.max_total_tokens,
                "user_prompt": query_param.user_prompt or "",
                "enable_rerank": query_param.enable_rerank,
            }
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=response,
                    prompt=query,
                    mode=query_param.mode,
                    cache_type="query",
                    queryparam=queryparam_dict,
                ),
            )

    # Return unified result based on actual response type
    if isinstance(response, str):
        # Non-streaming response (string)
        if len(response) > len(sys_prompt):
            response = (
                response[len(sys_prompt) :]
                .replace(sys_prompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .strip()
            )

        return QueryResult(content=response, raw_data=raw_data)
    else:
        # Streaming response (AsyncIterator)
        return QueryResult(
            response_iterator=response, raw_data=raw_data, is_streaming=True
        )

# 包含文档处理和查询的核心算法：

# chunking_by_token_size：文档分块函数
# extract_entities：实体提取函数
# merge_nodes_and_edges：节点和边的合并逻辑
# kg_query：知识图谱查询函数
# naive_query：朴素查询函数