from pathlib import Path
from typing import List, Dict, Tuple, Optional

def load_video_metadata(base_dataset_path: Path) -> Optional[List[Dict[str, str]]]:
    """
    Traverses the dataset directory structure (train/eval/test) and returns a list 
    of metadata dictionaries for each video, including its path, label (as string), and split.
    Its ONLY responsibility is to read from the file system.

    Args:
        base_dataset_path (Path): Path to the root 'dataset' directory.

    Returns:
        Optional[List[Dict[str, str]]]: List of metadata dictionaries, or None if an error occurs.
    """
    print(f"--- Cargando Metadatos desde: {base_dataset_path} ---")
    structured_data = []

    for split_name in ["train", "eval", "test"]:
        split_dir = base_dataset_path / split_name
        if not split_dir.is_dir():
            print(f"Advertencia: No se encontró el directorio {split_dir}")
            continue

        print(f"  Procesando: {split_dir}...")
        found_count = 0
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                label = class_dir.name
                for video_file in class_dir.glob("*.avi"): # O .mp4, etc.
                    structured_data.append({
                        "video_path": str(video_file.resolve()),
                        "label_str": label,
                        "split": split_name 
                    })
                    found_count += 1
        print(f"    -> {found_count} videos cargados.")

    if not structured_data:
        print("Error: No se encontraron videos.")
        return None
        
    print(f"Metadatos cargados para {len(structured_data)} videos.")
    return structured_data

def create_label_mappings(metadata: List[Dict[str, str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Takes the metadata and creates the label2id and id2label mappings.
    Its ONLY responsibility is to handle label management.
    
    Args:
        metadata (List[Dict[str, str]]): The list generated by `load_video_metadata`.
    
    Returns:
        Tuple[Dict[str, int], Dict[int, str]]: The label2id and id2label dictionaries.
    """
    print("\n--- Creando Mapeos de Etiquetas ---")
    all_labels_set = set(d["label_str"] for d in metadata)
    class_names = sorted(list(all_labels_set))
    
    label2id = {name: i for i, name in enumerate(class_names)}
    id2label = {i: name for i, name in enumerate(class_names)}
    
    print(f"Se crearon mapeos para {len(class_names)} clases.")
    return label2id, id2label

