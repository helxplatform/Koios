import aiohttp
import asyncio


async def resolve_to_identifiers_sapbert(concepts):
    concept_mapping = {}
    async with aiohttp.ClientSession() as session:
        tasks = []
        for concept in concepts:
            url = "https://sap-qdrant.apps.renci.org/annotate/"
            payload = {
                "text": concept,
                "model_name": "sapbert",
                "count": 1
            }
            tasks.append(session.post(url, json=payload))
        responses = await asyncio.gather(*tasks)
        for concept, r in zip(concepts, responses):
            if r.status == 200:
                response_json = await r.json()
                concept_mapping[concept] = response_json[0]['curie']
    return concept_mapping
