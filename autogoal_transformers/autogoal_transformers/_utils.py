import json
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import HfApi, ModelFilter 
import re
from enum import Enum

import requests

class DOWNLOAD_MODE(Enum):
    HUB = "hub"
    SCRAP = "scrap"
    
class ModelDescriptor():
    def __init__(self, modelId, downloads, likes, pipeline_tag):
        self.modelId = modelId
        self.downloads = downloads
        self.likes = likes
        self.pipeline_tag = pipeline_tag

def get_hf_models(target_task):
    hf_api = HfApi()
    return hf_api.list_models(filter=ModelFilter(task = target_task, library="pytorch"))

def get_hf_models_sorted_by_likes(target_task, min_likes, min_downloads):
    from bs4 import BeautifulSoup
    page = 0
    count = 0
    
    while True:
        url = f"https://huggingface.co/models?pipeline_tag={target_task}&sort=likes"
        if page > 0:
            url += f"&p={page}"

        response = requests.get(url)
        
        soup = BeautifulSoup(response.content.decode('utf8'))

        for model in soup.find_all('article'):

            parsed_text = [line.strip() for line in re.sub(' +', ' ', model.text.replace('\n', ' ').replace('\t', ' ').replace('•', '\n')).strip().split('\n')]
            model_name_str, last_updated_str, downloaded, *likes = parsed_text
            likes = int(likes[0]) if likes else 0
            downloads = convert_string_to_number(downloaded.strip())
            
            if (downloads < min_downloads):
                return
            
            if (likes < min_likes):
                return

            model_name = model.find('a').attrs['href'][1:]
            timestamp = model.find('time').attrs['datetime']
            yield ModelDescriptor(model_name, downloads, likes, target_task)

            count += 1
        page += 1



def get_model_config(modelId):
    config = AutoConfig.from_pretrained(modelId)
    return config


def get_models_info(target_task, max_amount, min_likes=None, min_downloads=None, download_mode=DOWNLOAD_MODE.HUB):
    models = get_hf_models(target_task.value) \
        if download_mode == DOWNLOAD_MODE.HUB \
        else get_hf_models_sorted_by_likes(target_task.value, min_likes, min_downloads)
    
    # regex for detecting partially trained models
    pattern = r"train-\d+"

    try:
        import progressbar

        # setup progress bar
        bar = progressbar.ProgressBar(
            maxval=1000,
            widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
        )

        # Start the progress bar
        bar.start()
    except:
        pass

    # Get model metadata
    model_info = []
    current = 0
    for model in models:
        if current >= max_amount:
            break

        if re.search(pattern, model.modelId) is not None:
            continue

        try:
            if download_mode == DOWNLOAD_MODE.SCRAP:
                likes, downloads = model.likes, model.downloads
            else:
                likes, downloads = get_model_likes_downloads(model.modelId)
            
            if (min_likes is not None and likes < min_likes):
                continue 
            
            if (min_downloads is not None and downloads < min_downloads):
                continue 
            
            config = get_model_config(model.modelId)

            info = {
                "name": model.modelId,
                "metadata": {
                    "task": model.pipeline_tag,
                    "id2label": config.id2label,
                    "model_type": config.model_type,
                    "likes": likes,
                    "downloads": downloads
                },
            }
            model_info.append(info)
            current += 1
            
            try:
                bar.update(current)
            except:
                pass
        except Exception as e:
            pass

    # Finish the progress bar
    bar.finish()
    return model_info


def download_models_info(
    target_task, 
    output_path="text_classification_models_info.json", 
    max_amount=1000, 
    min_likes=100, 
    min_downloads=1000,
    download_mode=DOWNLOAD_MODE.HUB
):
    # Get model info and dump to JSON file
    model_info = get_models_info(target_task, max_amount, min_likes, min_downloads, download_mode=download_mode)
    with open(output_path, "w") as f:
        json.dump(model_info, f)

    print(f"Model information has been saved to {output_path}.")
    return model_info


def get_model_likes_downloads(model_name):
    url = f"https://huggingface.co/{model_name}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the HTML element with the likes
    likes_element = soup.find('button', {'title': 'See users who liked this repository'})
    likes = int(likes_element.text) if likes_element else 0

    # Find the HTML element with the downloads
    downloads_element = soup.find('dt', text='Downloads last month').find_next_sibling('dd')
    downloads = int(downloads_element.text.replace(',', '')) if downloads_element else 0

    return likes, downloads


def to_camel_case(name):
    # Remove numbers at the beginning, replace '/' with '_', and split on '-'
    words = re.sub(r"^[0-9]*", "", name.replace("/", "_").replace(".", "")).split("-")
    return "".join(re.sub(r"^[0-9]*", "", word).title() for word in words)

def convert_string_to_number(s):
    """
    Convert a string to a number, where the string can end in 'k' or 'M' to signify thousands or millions.
    """
    units = {'k': 1000, 'M': 1000000}
    if s[-1] in units:
        return float(s[:-1]) * units[s[-1]]
    else:
        return float(s)
