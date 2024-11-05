from falkordb.asyncio import FalkorDB


class RedisGraphDB:
    def __init__(self, config):
        self.db = FalkorDB(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            password=config.REDIS_PASSWORD
        )
        self.graph = self.db.select_graph(config.REDIS_GRAPH_NAME)

    async def query_graph(self, cypher_str: str):
        return await self.graph.query(cypher_str)
