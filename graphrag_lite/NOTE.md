
pipeline file  
index workflows factory pipelineFactory: create_pipeline

config / chunking_strategy, search_mode, extract_graph_method, 
pipeline 省去 create_community_reports_text


Chunking后的页码：\n%PAGE0000%%PAGE0000%\n加在每页开头，chunking后识别最开头的%PAGE0000%

integrated : 多知识库的模式 如何做权限管理
搜索出多个，然后判断source_id (chunk_id)是不是在知识库里

所有文件用一个VectorStore存储
权限管理使用SQLite管理（定义KnowledgeBase类，包括kbId，userId，file_id，chunk_id etc.）

删除逻辑：删掉所有chunk_id
vector_store 
- Chunk 直接按照chunk_id删
- Entity 删掉chunk_id in source_id，如果source_id变为空则删除这个Entity
- Relationship 删掉chunk_id in source_id，source_id变为空则删除这个Relationship
- 删除完后 


vector_store lancedb vs milvus
graph local graphml remote NebulaGraph
community : use report or not


# custom llm and embedding parameters
修改了openai依赖的代码
openai: open.resources.embeddings:AsyncEmbeddings.create :220
params里加入`"isNorm": False`

index
query
llm

use langchain v0.3.24
```
├── knowledge_base_management

├── light_graphrag
|   ├── data_core (entity, relationship, text_unit, ...)
|   ├── _typing
|       ├── types.py 
|       ├── enums.py 
|   ├── config
|       ├── graphrag_config
|       ├── index_config
|       ├── query_config

|   ├── index
|       ├── op
|           ├── chunk_text
|           ├── extract_graph
|           ├── embed_graph

|       ├── run
|           ├── run_chain

|   ├── query
|       ├── op
|           ├── context_build
|           ├── question_reform 
|           ├── local_search
|   ├── prompts

|   ├── llm
|       ├── types.py (input, response, model_parameters)
|   ├── db vector
|   ├── storage


|   +-- c
```
流程
run_chunk_text

run_extract_graph_nlp : using nlp to get NP, pruning graph, store graph

新建知识库 pdf -> text 存到input --chunking-> text_unit 
"create_base_text_units",
"create_final_documents",
"extract_graph_nlp",
"prune_graph",
"finalize_graph",
"create_communities",
"create_final_text_units",
"create_community_reports_text",
"generate_text_embeddings",

with_structured_output
定义run_* callable 的 格式 config, RunContext -> None
RunContext [Cache, Storage, Callback(主要处理progress)]
run_chunk_text -> get TextChunk & documents -> save to storage

- load_storage
- run ...
- write_storage

run_extract_graph_nlp -> extract, prune, finalize (store) graph

RunOp
callable[[config, RunContext], None]

run_op文件里 有一个类定义了RunOp, 全过程Run, 包含了run_op的所有过程

RunStorage
查询逻辑
default_tenantId = "VIRTUAL_TENANT_ID"
default_userId = "VIRTUAL_USER_ID"
按照用户来存，working_dir = "#(tenantId)##(userId)###"

RunContext Cache(langchain) RunStorage RunStats RunCallback

要考虑到往NebulaGraph里INSERT的数据结构



index和query对数据库操作 互斥机制

存储结构 

entity，relationship，text_unit 按知识库分别存储

删除逻辑：对应的知识库里的 entity，relationship，text_unit 直接删除



多个知识库问答：将entity和relationship 的 df 拼接起来

Lancedb

Hybrid Search
https://docs.lancedb.com/core/hybrid-search


LLM 
retry strategy
Exponential Backoff with Jitter
https://apxml.com/courses/prompt-engineering-llm-application-development/chapter-7-output-parsing-validation-reliability/implementing-retry-mechanisms

> Here's the logic:
> 1. Try the operation.
> 2. If it fails, wait for a base delay (e.g., 1 second).
> 3. If it fails again, wait for base_delay * 2 + random_jitter.
> 4. If it fails again, wait for base_delay * 4 + random_jitter.
> 5. Continue doubling the wait time (up to a reasonable maximum delay) until the maximum number of retries is reached.


Langchain-Chain
https://python.langchain.com/docs/tutorials/sql_qa/#chains


Langchain-Context
https://api.python.langchain.com/en/latest/core/beta/langchain_core.beta.runnables.context.Context.html

SQLStore InMemoryStore
()

Langchain chain with storage and cache

Langchain-Store

Run Operation with Langchain RunnableLambda
https://python.langchain.com/docs/how_to/functions/

Langchain Callbacks and Custom Events 
https://python.langchain.com/docs/how_to/callbacks_custom_events/
https://python.langchain.com/docs/how_to/streaming/#propagating-callbacks


store-
tenant -> User -> [knowledgeBase1, knowledgeBase2, ..., ]

问答 
https://langchain-ai.github.io/langgraph/how-tos/#persistence