import config
import json


def get_study_data(study_id, comparator=lambda x, y: x == y, exclude_keys=None):
    # @TODO replace this with some db....
    with open(config.STUDIES_JSON_FILE) as stream:
        data = json.load(stream)
    for study in data:
        if comparator(study['StudyId'], study_id):
            data = {
                "study_id": study['StudyId'],
                "study_name": study['StudyName'],
                "permalink": study['Permalink'],
                "description": study['Description']
            }
            if exclude_keys is not None:
                data = {x: data[x] for x in data if x not in exclude_keys}
            return data, 200

    default_data = {"study_name": "", "permalink": "", "description": "", "study_id": ""}
    if exclude_keys is not None:
        default_data = {x: "" for x in default_data if x not in exclude_keys}
    return default_data, 404
