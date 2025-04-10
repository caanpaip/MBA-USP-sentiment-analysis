import emoji
from pandas import DataFrame
import typing_extensions
import numpy as np
import pandas as pd
from sklearn.metrics import  classification_report
from scipy.sparse import spmatrix

ArrayLike = np.typing.ArrayLike
MatrixLike: typing_extensions.TypeAlias = np.ndarray | pd.DataFrame | spmatrix

def emojis_extractions(text: str):
    """Extracts emojis from a given text.

    Parameters
    ----------
    text : str
        The text to extract emojis from.

    Returns
    -------
    list
        A list containing all emojis found in the text.

    Examples
    --------
    >>> emojis_extractions("Hello world! 👋")
    ['👋']

    >>> emojis_extractions("This is a text with no emojis.")
    []
    """
    emoji_ls = []
    for char in text:
        if emoji.is_emoji(char):
            emoji_ls.append(char)

    return emoji_ls


def pricing(
    df: DataFrame,
    columns_cost: list = [
        "input_$_gpt_35",
        "input_$_gpt_4o",
        "input_$_gpt_4_turbo",
        "input_$_gpt_4",
        "input_$_gpt_4o-mini",
        
    ],
):
    """Calculates the total cost of each model and the overall total cost.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing columns representing the cost of each model per token.
    columns_cost : list, optional
        A list of column names in the DataFrame that represent the cost per token
        for each model. The default value is
        ['input_$_gpt_35', 'input_$_gpt_4o', 'input_$_gpt_4_turbo', 'input_$_gpt_4'].

    Returns
    -------
    tuple[pd.DataFrame, float]
        A tuple containing two elements:
            - A DataFrame that groups the costs by model.
            - The overall total cost.

    Examples
    --------
    >>> sample_10 = pd.DataFrame({'input_$_gpt_35': [1, 2, 3],
    ...                         'input_$_gpt_4o': [4, 5, 6],
    ...                         'input_$_gpt_4_turbo': [7, 8, 9],
    ...                         'input_$_gpt_4': [10, 11, 12]})
    >>> pricing_by_model, total_cost = pricing(sample_10)
    >>> pricing_by_model
    modelo
    input_$_gpt_35         6.0
    input_$_gpt_4o        15.0
    input_$_gpt_4_turbo    24.0
    input_$_gpt_4         33.0
    Name: tokens ($USD), dtype: float64
    >>> total_cost
    tokens ($USD)    78.0
    dtype: float64

    """

    df_pricing = df[columns_cost].melt(var_name="modelo", value_name="tokens ($USD)")

    pricing_by_model = df_pricing.groupby("modelo").agg("sum")
    pricing_by_model["lines"] = df.shape[0]

    total_cost = pricing_by_model.sum()

    return pricing_by_model, total_cost


def remove_empty_elements(lista: list)-> list:
    """
    Função que remove elementos vazios de uma lista
    Args:
        lista (list): _description_

    Returns:
        _type_: _description_
    """
    return [x for x in lista if x!=""]


def classification_report_table( y_true: MatrixLike | ArrayLike, y_pred: MatrixLike | ArrayLike, num_classes: int  )-> DataFrame:
    
    """Gera a tabela de classificação a partir do classification_report
    Args:
        y_true (MatrixLike | ArrayLike): true labels
        y_pred (MatrixLike | ArrayLike): pred labels
        num_classes (int): number of classes
    Returns:
        df_metrics (pd.DataFrame): dataframe with the classification report
    """

    # Generate a classification report and process it to extract metrics labels.
    metrics = classification_report( y_true, y_pred)
    # Split the report into lines and remove empty lines.
    values_list = remove_empty_elements(metrics.split("\n"))
    # Extract the lines corresponding to the class labels and the overall metrics.
    metrics_labels = values_list[:num_classes+1]
    # Add "labels" to the beginning of the first label.
    metrics_labels[0] = "labels" + metrics_labels[0]
    # Split each label line into individual elements, removing empty strings.
    metrics_labels = [remove_empty_elements( x.split(" ") ) for x in metrics_labels]
    # The resulting list of lists containing the processed metric labels.
    df_metrics = pd.DataFrame(metrics_labels[1:], columns=metrics_labels[0])

    return df_metrics


def version_label(model: str, n:int)->str:
    """
    Função que gera os labels para as versões do modelo

    Args:
        model (str): model
        n (int): number of execution
    Returns:
        str: version label in execution n
    """

    if n==0:
        if model=="gpt-3.5-turbo":
            return "0613"
        elif model=="gpt-4o":
            return "2024-05-13"       
    elif n==1:
        if model=="gpt-3.5-turbo":
            return "0613" 
        elif model=="gpt-4o":
            return "2024-05-13"
        elif model=="gpt-4o-mini":
            return "2024-07-18"
    elif n==2:
        if model=="gpt-3.5-turbo":
            return "0125"
        elif model=="gpt-4o":
            return "2024-08-06"
        elif model=="gpt-4o-mini":
            return "2024-07-18"
        
        
class CompareResultsMetrics:
    
    def __init__(self, df:DataFrame):
        """
        Função para comparar resultados de diferentes execuções de um mesmo modelo e versão.
        """
        self.df = df 

    def struct_results(self, df:DataFrame, execs: list, version_labeling:bool = False )->DataFrame:
        
        """
        Função para comparar resultados de diferentes execuções de um mesmo modelo e versão.
        Args:
            df (DataFrame): DataFrame com os resultados das métricas de avaliação Filtrados.
            execs (list): Lista de execuções a serem comparadas.

        Returns:
            DataFrame: DataFrame filtrado com os resultados das execuções comparadas.
        """
    
        df_list = []

        for exec in execs:
            
            df_exec = df[df["execution"]==exec]
            
            if version_labeling:
                version = df_exec["version"].unique()[0]
                rename_cols = "{col} exec {exec} ({version})"
            else:
                version = ""
                rename_cols = "{col} exec {exec}"
            
            df_exec = df_exec.drop(columns=["model","version","prompt","execution","support"])
            
            colums = df_exec.columns.tolist()
            colums.remove("labels")
            
            df_exec.rename(columns={col: rename_cols.format( col = col, exec = exec, version = version) for col in colums}, inplace=True)

            df_list.append(df_exec)
            
        df_compare = df_list[0].merge(df_list[1], on="labels", how="inner")

        # Supondo que seu DataFrame seja chamado df
        columns = df_compare.columns.tolist()
        columns.remove("labels")

        # Ordenar as colunas de forma genérica
        sorted_columns = sorted(columns)
        columns = ["labels"] + sorted_columns[::-1]

        return df_compare[columns]


    def same_model_and_version(self, model:str, version:str, prompt:str)->DataFrame:
        """
        Função para comparar resultados de diferentes execuções de um mesmo modelo e versão.
        Args:
            model (str): Nome do modelo a ser filtrado.
            version (str): Versão do modelo a ser filtrada.
            prompt (str): Prompt a ser filtrado.
            
        Returns:
            DataFrame: DataFrame filtrado com os resultados das execuções comparadas.
        """

        df_same_model_and_version = self.df[(self.df["model"].str.contains(model, regex=False, na=False, case=False)) & (self.df["version"].str.contains(version, regex=False, na=False, case=False)) & (self.df["prompt"].str.contains(prompt, regex=False, na=False, case=False))]

        execs = df_same_model_and_version["execution"].unique().tolist()

        return self.struct_results(df_same_model_and_version, execs)
    
    
    def same_model_and_diff_version(self, model:str, prompt:str)->DataFrame:


        df_same_model_and_diff_version = self.df[(self.df['model']==model) & (self.df['prompt'].str.contains( prompt, regex=False, na=False, case=False))]
        ## two last executions  
        execs = sorted (df_same_model_and_diff_version["execution"].unique().tolist())[-2::]
        
        # logger.info(f"Execuções a serem comparadas: {execs}")
        
        return self.struct_results(df_same_model_and_diff_version, execs, True)
    

