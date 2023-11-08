import json
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import HfApi
import progressbar
import re


def get_hf_models(target_task):
    hf_api = HfApi()
    return hf_api.list_models(filter=target_task)


def get_model_config(modelId):
    config = AutoConfig.from_pretrained(modelId)
    return config


def get_models_info(target_task, max_amount):
    models = get_hf_models(target_task.value)

    # regex for detecting partially trained models
    pattern = r"train-\d+"

    # setup progress bar
    bar = progressbar.ProgressBar(
        maxval=1000,
        widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
    )

    # Start the progress bar
    bar.start()

    # Get model metadata
    model_info = []
    current = 0
    for model in models:
        if current >= max_amount:
            break

        if re.search(pattern, model.modelId) is not None:
            continue

        try:
            config = get_model_config(model.modelId)

            info = {
                "name": model.modelId,
                "metadata": {
                    "task": model.pipeline_tag,
                    "tags": model.tags,
                    "id2label": config.id2label,
                    "model_type": config.model_type,
                },
            }
            model_info.append(info)
            current += 1
            bar.update(current)
        except:
            pass

    # Finish the progress bar
    bar.finish()
    return model_info


def download_models_info(
    target_task, output_path="text_classification_models_info.json", max_amount=1000
):
    # Get model info and dump to JSON file
    model_info = get_models_info(target_task, max_amount)
    with open(output_path, "w") as f:
        json.dump(model_info, f)

    print(f"Model information has been saved to {output_path}.")
    return model_info


# Function to convert names to CamelCase
def to_camel_case(name):
    # Remove numbers at the beginning, replace '/' with '_', and split on '-'
    words = re.sub(r"^[0-9]*", "", name.replace("/", "_").replace(".", "")).split("-")
    return "".join(re.sub(r"^[0-9]*", "", word).title() for word in words)

# models = get_hf_models("zero-shot-classification")

# models = list(get_hf_models("token-classification"))
# print(models)

