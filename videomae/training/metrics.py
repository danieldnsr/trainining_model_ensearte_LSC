import numpy as np
import evaluate

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    """
    Calcula accuracy, F1, precision y recall para la evaluación.
    Utiliza promedios 'macro' y 'weighted' para F1, precision y recall.

    Args:
        eval_pred (EvalPrediction): Objeto que contiene las predicciones (logits)
                                    y las etiquetas verdaderas.

    Returns:
        dict: Un diccionario con los nombres de las métricas y sus valores.
    """
    # 1. Extraer logits y etiquetas
    logits, labels = eval_pred
    
    # 2. Convertir logits a predicciones (la clase con mayor probabilidad)
    predictions = np.argmax(logits, axis=1)

    # 3. Calcular las métricas
    acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    
    f1_macro = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
    f1_weighted = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    
    precision_macro = precision_metric.compute(predictions=predictions, references=labels, average="macro")["precision"]
    precision_weighted = precision_metric.compute(predictions=predictions, references=labels, average="weighted")["precision"]

    recall_macro = recall_metric.compute(predictions=predictions, references=labels, average="macro")["recall"]
    recall_weighted = recall_metric.compute(predictions=predictions, references=labels, average="weighted")["recall"]

    # 4. Devolver un diccionario con todas las métricas
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
    }
