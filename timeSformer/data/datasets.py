from typing import Dict, List, Optional
from datasets import Dataset, DatasetDict, ClassLabel, Features, Value

def build_huggingface_dataset(metadata: List[Dict[str, str]], label2id: Dict[str, int]) -> Optional[DatasetDict]:
    """
    Builds the Hugging Face DatasetDict from metadata and label mappings.
    Its SOLE responsibility is to create the DatasetDict object.
    
    Args:
        metadata (List[Dict[str, str]]): The list of metadata dictionaries.
        label2id (Dict[str, int]): The label-to-ID mapping dictionary.
    
    Returns:
        Optional[DatasetDict]: The 'raw' DatasetDict or None if an error occurs.
    """
    class_names = sorted(label2id.keys(), key=lambda x: label2id[x])
    class_label_feature = ClassLabel(names=class_names)
    features = Features({
        'video_path': Value('string'),
        'label': class_label_feature
    })

    datasets_dict_content = {}
    for split_name_hf in ["train", "validation", "test"]:
        # Mapea 'validation' a 'eval' si es necesario para buscar en tus metadatos
        split_to_search = "eval" if split_name_hf == "validation" else split_name_hf
        
        split_data = [d for d in metadata if d["split"] == split_to_search]

        if not split_data:
            print(f"Advertencia: No hay datos para el split '{split_name_hf}'.")
            continue

        datasets_dict_content[split_name_hf] = Dataset.from_dict({
            "video_path": [d["video_path"] for d in split_data],
            "label": [label2id[d["label_str"]] for d in split_data]
        }, features=features)

    if not datasets_dict_content:
        print("Error: No se pudo crear ningún dataset split.")
        return None

    raw_dataset = DatasetDict(datasets_dict_content)
    print("¡DatasetDict construido!")
    return raw_dataset