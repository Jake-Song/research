from pprint import pprint

from huggingface_hub import HfApi, get_space_runtime

api = HfApi()

# Get Space runtime info
runtime = get_space_runtime("Jakemu/openspiel_env")
print(runtime)

# Get Space variables and secrets
space_vars = api.get_space_variables(repo_id="Jakemu/openspiel_env")
print("Space variables:")
pprint(space_vars)