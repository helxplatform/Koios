from typing import Any
from langchain_community.vectorstores import Qdrant
from langchain_community.docstore.document import Document
from util.study_data import get_study_data



class CustomQdrant(Qdrant):
    """
    Class performs lookup on the Qdrant in chains.
    """
    @classmethod
    def _document_from_scored_point(
            cls,
            scored_point: Any,
            collection_name: str,
            content_payload_key: str,
            metadata_payload_key: str,
    ) -> Document:
        """
        This method is overriden to get the documents form a local file to provide for the context.
        :param scored_point:
        :param collection_name:
        :param content_payload_key:
        :param metadata_payload_key:
        :return:
        """
        metadata = scored_point.payload.get(metadata_payload_key) or {}
        metadata["_id"] = scored_point.id
        metadata["_collection_name"] = collection_name
        study_id = scored_point.payload.get('question_id').split('.')[0]
        metadata["study_id"] = study_id
        metadata["score"] = scored_point.score
        raw_study, status = get_study_data(study_id)
        if status == 200:
            page_content = f"{raw_study['study_name']} ({raw_study['study_id']}): \n {raw_study['description']}"
        else:
            page_content = ""
        return Document(
            page_content=page_content,
            metadata=metadata,
        )
