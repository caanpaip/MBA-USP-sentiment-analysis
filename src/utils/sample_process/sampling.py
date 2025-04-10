from pandas import DataFrame


class Sampling:
    def __init__(self, df: DataFrame, perc: float = 0.1, k: int = 1, random_state: int = 0):
        self.df = df
        self.perc = perc
        self.k = k
        self.random_state = random_state

    @staticmethod
    def __sampling(df: DataFrame, perc: float, k: int, random_state: int):
        """
        Returns a sample of a DataFrame.

        Parameters
        ----------
        df : DataFrame
            The DataFrame to be sampled.
        perc : float
            The percentage of the sample to be taken.
        k : int, optional
            The minimum number of samples to be taken. Defaults to 1.
        random_state : int, optional
            The random state to be used. Defaults to 0.

        Returns
        -------
        pd.DataFrame
            The sample of the DataFrame.
        """

        n = max(k, int(len(df) * perc))
        return df.sample(n=n, random_state=random_state)

    def Simple_sampling(self):
        """
        Returns a sample of a DataFrame.

        Returns
        -------
        pd.DataFrame
            The sample of the DataFrame.
        """

        return self.__sampling(self.df, self.perc, self.k, self.random_state)

    def stratified_sample(self, column: str):
        """
        Returns a stratified sample of a DataFrame.

        Parameters
        ----------
        Column : str
            Column name to stratify


        Returns
        -------
        pd.DataFrame
            The stratified sample of the DataFrame.
        """

        df_stratify = self.df.groupby(column, group_keys=False).apply(
            lambda x: self.__sampling(x, self.perc, self.k, self.random_state)
        )

        return df_stratify
