import redis
from redisgraph import Graph

class KGAgent:
    def __init__(self, local_kg=None, redis_host='localhost', redis_port=6379, redis_password="NbzI3cp3uT", graph_name='your_graph_name'):
        self.local_kg = local_kg if local_kg else {}
        self.redis_client = redis.StrictRedis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=True  # Ensure that responses are decoded to strings
        )
        self.graph = Graph(graph_name, self.redis_client)  # Initialize the RedisGraph object

    def query_entity(self, entity):
        # Combine local KG lookup with Redis lookup
        related_entities, hidden_insights = self.lookup_kg_with_insights([entity])
        return related_entities, hidden_insights

    
    def lookup_kg_with_insights(self, entities):
        subgraph = {}
        hidden_insights = {
            "hidden_relationships": [],
            "important_entities": [],
            "hidden_concepts": []
        }

        for entity in entities:
            if entity in self.local_kg:
                subgraph[entity] = self.local_kg[entity]
            else:
                data = self.lookup_in_redis(entity)
                if data:
                    subgraph[entity] = data

            if entity in subgraph:
                print(f"Subgraph for {entity}: {subgraph[entity]}")  # Debug print

                for concept in subgraph[entity].get("related_concepts", []):
                    if concept not in entities and concept not in subgraph:
                        hidden_insights["hidden_concepts"].append(concept)
                    if concept in entities:
                        hidden_insights["important_entities"].append(concept)

                for study in subgraph[entity].get("related_studies", []):
                    if study not in hidden_insights["hidden_relationships"]:
                        hidden_insights["hidden_relationships"].append(study)

        print(f"Hidden insights: {hidden_insights}")  # Debug print
        return subgraph, hidden_insights

    def lookup_in_redis(self, entity):
        query = f"""
        MATCH (a {{id: '{entity}'}})-[e1]->(n1)-[e2]->(b:`biolink.StudyVariable`)
        RETURN a.id, n1.name, b.name LIMIT 10
        """
        result = self.graph.query(query)
        
        subgraph = {
            'related_concepts': [],
            'related_studies': []
        }
        
        for record in result.result_set:
            n1_name = record[1]  # Second column is the name of the related concept
            b_name = record[2]   # Third column is the name of the study
            
            subgraph['related_concepts'].append(n1_name)
            subgraph['related_studies'].append(b_name)
        
        return subgraph if subgraph['related_concepts'] or subgraph['related_studies'] else None

    # def lookup_in_redis(self, entity):
    #     # Perform a Cypher query in RedisGraph to get the subgraph for the entity
    #     query = f"""
    #     MATCH (a {{id: '{entity}'}})-[r]->(b)
    #     RETURN a, r, b
    #     """
    #     result = self.graph.query(query)
        
    #     subgraph = {
    #         'nodes': [],
    #         'relationships': []
    #     }
        
    #     # Extract nodes and relationships from the query result
    #     for record in result.result_set:
    #         a_node = record[0]
    #         relationship = record[1]
    #         b_node = record[2]
            
    #         subgraph['nodes'].append(a_node)
    #         subgraph['nodes'].append(b_node)
    #         subgraph['relationships'].append(relationship)
        
    #     return subgraph if subgraph['nodes'] else None

    def chain(self, query_entity):
        related_entities, hidden_insights = self.query_entity(query_entity)
        return related_entities, hidden_insights


