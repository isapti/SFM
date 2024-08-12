import base64
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

from utils import logger
from utils.config_loader import config

# %% Setup logger
logger = logger.get_logger(__name__)
# %% Setup test data directories
ROOT_DIR = Path(".")
IMPORT_DATA_DIR = ROOT_DIR / "data" / "external"
MAPPING_DIR = ROOT_DIR / "data" / "mapping"
RULES_DIR = ROOT_DIR / "data" / "rules"

# %% Load configuration from file
token_url = config["access_token_config"]["token_url"]
client_id = config["access_token_config"]["client_id"]
client_secret = config["access_token_config"]["client_secret"]
username = config["access_token_config"]["username"]
password = config["access_token_config"]["password"]
project_id = config["rule_properties"]["project_id"]
message_classification_id = config["rule_properties"]["message_classification_id"]
schema_name = config["rule_properties"]["schema_name"]

get_all_rules = config["rules"]["get_all_rules"]
get_rule_by_id = config["rules"]["get_rule_by_id"]

# %% Generic HTTP request function
def generic_request(request_method, endpoint, headers, payload):
    """
    Perform a generic HTTP request.

    Args:
        request_method (str): The HTTP method (e.g., "GET", "POST").
        endpoint (str): The endpoint URL.
        headers (dict): Request headers.
        payload (dict): Request payload.

    Returns:
        requests.Response: The response object.

    Raises:
        requests.exceptions.RequestException: If an error occurs during the request.
    """
    try:
        response = requests.request(
            request_method,
            endpoint,
            headers=headers,
            data=json.dumps(payload),
            timeout=20,
            verify=False,
        )
        response.raise_for_status()  # Raise an exception if the response status is not in the 200 range
        return response
    except requests.exceptions.RequestException as e:
        # Log the error or handle it as required
        logger.error(e)
        # Return a default response
        default_response = requests.Response()
        default_response.status_code = 500
        default_response._content = b'{"message": "Internal Server Error"}'
        return default_response

######################################################## ACCESS TOKEN #################################################

# Requesting OAuth2 token
try:
    token_response = requests.post(
        token_url,
        auth=HTTPBasicAuth(client_id, client_secret),
        data={
            "grant_type": "password",
            "username": username,
            "password": password,
            "scope": "openid",
        },
        verify=False,  # No SSL
        timeout=10,
    )
    # Extracting the access token from the response
    access_token = token_response.json().get("access_token")
except requests.exceptions.RequestException as e:
    access_token = ""
    logger.error(e)

def replace_variables(input_string, mapping, prefix):
    for old_var, new_var in mapping.items():
        if new_var:  # Check if new variable name is not blank
            new_var_with_prefix = f"{prefix}.{new_var}"
            input_string = re.sub(
                r"\b{}\b".format(re.escape(old_var)),
                new_var_with_prefix,
                input_string,
                flags=re.IGNORECASE,
            )
    return input_string

def read_mapping_from_csv(csv_file):
    mapping = {}
    with open(MAPPING_DIR / csv_file, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            old_var = row[0].strip()
            new_var = row[1].strip()
            mapping[old_var] = new_var
    return mapping

################################################### READ SFM RULES FROM JSON #################################################
def read_rules_from_file():
    file_path = RULES_DIR / "SFM_rules.json"
    with open(file_path, "r", encoding="utf-8") as file:
        rules_data = json.load(file)
    return rules_data

class Rule_SFM:
    all_rules = []

    def __init__(self, json_data):
        for key, value in json_data.items():
            setattr(self, key, value)

        self.code = self.code.replace("&rule.", "")

        pattern = re.compile(r"%ListContains\((.*?),(.*?)\)")
        replacer = (
            lambda match: f"(Lists.{match.group(1).strip()}.contains({match.group(2).strip()}))"
        )
        self.code = pattern.sub(replacer, self.code)

        pattern = re.compile(
            r"dhms\(\s*RQO_TRAN_DATE\s*,\s*0\s*,\s*0\s*,\s*RQO_TRAN_TIME\s*\)", re.IGNORECASE
        )
        self.code = re.sub(pattern, "message.request.messageDtTm", self.code)
        pattern = re.compile(
            r"dhms\(\s*RQO_TRAN_DATE_ALT\s*,\s*0\s*,\s*0\s*,\s*RQO_TRAN_TIME_ALT\s*\)",
            re.IGNORECASE,
        )
        self.code = re.sub(pattern, "message.request.messageDtTm", self.code)

        pattern = r"%Action_Return_Result\(([^,]+),([^)]+)\);"
        replacement = r"/* Action_Return_Result(\1,\2); COMMENTED ACTION_RETURN_RESULT MACRO */"
        self.code = re.sub(pattern, replacement, self.code)

        pattern = r"%set\s*\(([^)]*)\)"
        self.code = re.sub(pattern, r"\1", self.code)

        self.name = self.name.replace("[", "").replace("]", "")
        self.name = self.name.replace("'", "")
        self.name = self.name.replace("/", "")
        self.name = self.name.replace("\\", "")
        self.code = self.code.replace("~", "")
        self.code = self.code.replace(" EQ ", " eq ")
        self.code = self.code.replace(" NE ", " ^= ")

        self.code = re.sub(
            r"%ShiftHistoryArray\(([^)]+)\)",
            r"_PLACEHOLDER_SHIFTHISTORYARRAY_(\1)",
            self.code,
            flags=re.IGNORECASE,
        )
        self.code = re.sub(
            r"%DeclareArray\(([^)]+)\)",
            r"_PLACEHOLDER_DECLAREARRAY_(\1)",
            self.code,
            flags=re.IGNORECASE,
        )
        self.code = re.sub(
            r"%indexArray\(([^)]+)\)",
            r"_PLACEHOLDER_INDEXARRAY_(\1)",
            self.code,
            flags=re.IGNORECASE,
        )
        self.code = re.sub(
            r"%get\(([^)]+)\)", r"_PLACEHOLDER_GET_(\1)", self.code, flags=re.IGNORECASE
        )

        pattern = r"%Action_([^;()]+)(;|\(\);)"
        replacement = r"detection.\1();"
        self.code = re.sub(pattern, replacement, self.code)

        self.code = re.sub(r"%Action_(.*?)(?:\(\w*\))?", r"detection.\1", self.code)

        sfm_api_mappings_path = "SFM_API_mappings.csv"
        sfm_uv_mappings_path = "SFM_UV_mappings.csv"

        sfm_api_mappings = read_mapping_from_csv(sfm_api_mappings_path)
        self.code = replace_variables(self.code, sfm_api_mappings, "message")

        sfm_uv_mappings = read_mapping_from_csv(sfm_uv_mappings_path)
        self.code = replace_variables(self.code, sfm_uv_mappings, "profile")

        Rule_SFM.all_rules.append(self)

    def replace_variables_in_string(input_string, variable_mapping):
        for old_variable, new_variable in variable_mapping.items():
            input_string = input_string.replace(old_variable, new_variable)
        return input_string

    def create_rule_json(self, messageClassificationId, schemaName):
        json_dict = {}
        json_dict["name"] = self.name
        json_dict["description"] = self.desc
        json_dict["messageClassificationId"] = messageClassificationId
        setattr(self, "messageClassificationId", messageClassificationId)
        json_dict["schemaName"] = schemaName
        setattr(self, "schemaName", schemaName)

        if self.rule_type == "Variable":
            json_dict["ruleTypeName"] = "variable"
        elif self.rule_type == "Authorization" or self.rule_type == "Queue":
            json_dict["ruleTypeName"] = "decision"
            json_dict["alertType"] = self.alert_type
            json_dict["alertReason"] = self.alert_reason

        setattr(self, "created_rule_json", json.dumps(json_dict, indent=2))
        return self.created_rule_json

    def update_rule_json(self):
        json_dict = {}
        json_dict["revision"] = 0
        json_dict["name"] = self.name
        json_dict["description"] = self.desc
        json_dict["schemaName"] = self.schemaName

        json_dict["code"] = base64.b64encode(self.code.encode("utf-8")).decode("utf-8")

        json_dict["operationalTimeLimit"] = 20

        if self.rule_type == "Variable":
            json_dict["ruleTypeName"] = "variable"
        elif self.rule_type == "Authorization" or self.rule_type == "Queue":
            json_dict["ruleTypeName"] = "decision"
            json_dict["alertType"] = self.alert_type
            json_dict["alertReason"] = self.alert_reason

        setattr(self, "updated_rule_json", json.dumps(json_dict, indent=2))
        return self.updated_rule_json

    def save_rule_to_file(self):
        file_name = self.name.replace(" ", "_") + ".json"
        file_path = RULES_DIR / file_name
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(self.updated_rule_json)

################################################### SFD RULES #################################################
def get_rule_by_id_request(rule_id):
    """
    Get a rule by its ID.

    Args:
        rule_id (str): The ID of the rule.

    Returns:
        dict: The rule information.
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    response = generic_request("GET", get_rule_by_id.format(rule_id), headers, None)
    if response.status_code == 200:
        rule_info = response.json()
        return rule_info
    else:
        logger.error(f"Failed to get rule by ID {rule_id}. Status code: {response.status_code}")
        return {}

def post_rule_request(payload):
    """
    Post a new rule.

    Args:
        payload (dict): The rule data to post.

    Returns:
        dict: The response from the server.
    """
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    response = generic_request("POST", get_all_rules, headers, payload)
    if response.status_code == 200:
        return response.json()
    else:
        logger.error(f"Failed to post rule. Status code: {response.status_code}")
        return {}

def update_rule_request(rule_id, payload):
    """
    Update an existing rule.

    Args:
        rule_id (str): The ID of the rule.
        payload (dict): The updated rule data.

    Returns:
        dict: The response from the server.
    """
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    response = generic_request("PUT", get_rule_by_id.format(rule_id), headers, payload)
    if response.status_code == 200:
        return response.json()
    else:
        logger.error(f"Failed to update rule {rule_id}. Status code: {response.status_code}")
        return {}

def delete_rule_request(rule_id):
    """
    Delete a rule by its ID.

    Args:
        rule_id (str): The ID of the rule.

    Returns:
        dict: The response from the server.
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    response = generic_request("DELETE", get_rule_by_id.format(rule_id), headers, None)
    if response.status_code == 204:
        return {"message": "Rule deleted successfully"}
    else:
        logger.error(f"Failed to delete rule {rule_id}. Status code: {response.status_code}")
        return {}

def create_rule_in_sfd(rule):
    """
    Create or update a rule in SFD.

    Args:
        rule (Rule_SFM): The rule to create or update.

    Returns:
        dict: The response from the server.
    """
    payload = rule.create_rule_json(message_classification_id, schema_name)
    existing_rule = get_rule_by_id_request(rule.name)
    if existing_rule:
        # Update existing rule
        rule_id = existing_rule.get("id")
        update_payload = rule.update_rule_json()
        return update_rule_request(rule_id, update_payload)
    else:
        # Create new rule
        return post_rule_request(payload)

def process_rules():
    rules_data = read_rules_from_file()
    for rule_data in rules_data:
        rule = Rule_SFM(rule_data)
        create_rule_in_sfd(rule)

if __name__ == "__main__":
    process_rules()
