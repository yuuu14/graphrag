# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

from graphrag.config.create_graphrag_config import create_graphrag_config
from graphrag.index.workflows.extract_graph_nlp import (
    run_workflow,
)
from graphrag.utils.storage import load_table_from_storage
from tests.verbs.util import (
    DEFAULT_MODEL_CONFIG,
    create_test_context,
)


async def test_extract_graph_nlp():
    context = await create_test_context(
        storage=["text_units"],
    )

    config = create_graphrag_config({"models": DEFAULT_MODEL_CONFIG})

    await run_workflow(config, context)

    nodes_actual = await load_table_from_storage("entities", context.storage)
    edges_actual = await load_table_from_storage("relationships", context.storage)

    nodes_actual = nodes_actual.sort_values(by="frequency", ascending=False)

    def convert_entities_to_dicts(df):
        """Convert the entities dataframe to a list of dicts for yfiles-jupyter-graphs."""
        nodes_dict = {}
        for _, row in df.iterrows():
            # Create a dictionary for each row and collect unique nodes
            node_id = row["title"]
            if node_id not in nodes_dict:
                nodes_dict[node_id] = {
                    "id": node_id,
                    "properties": row.to_dict(),
                }
        
        return list(nodes_dict.values())
    
    def convert_relationships_to_dicts(df):
        """Convert the relationships dataframe to a list of dicts for yfiles-jupyter-graphs."""
        relationships = []
        for _, row in df.iterrows():
            # Create a dictionary for each row
            relationships.append({
                "start": row["source"],
                "end": row["target"],
                "properties": row.to_dict(),
            })
        return relationships

    from pprint import pp
    _nodes_actual = nodes_actual[nodes_actual["frequency"] >= 5]
    entities_dicts = convert_entities_to_dicts(_nodes_actual)
    pp(entities_dicts)
    
    edges_actual = edges_actual.sort_values(by="weight", ascending=False)
    _edges_actual = edges_actual[edges_actual["weight"] >= 0.0001]
    relation_dicts = convert_relationships_to_dicts(_edges_actual)
    pp(relation_dicts)

    # this will be the raw count of entities and edges with no pruning
    # with NLP it is deterministic, so we can assert exact row counts
    # assert len(nodes_actual) == 1148
    # assert len(nodes_actual.columns) == 5
    # assert len(edges_actual) == 29445
    # assert len(edges_actual.columns) == 5
