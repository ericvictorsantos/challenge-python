from glob import glob
from time import time, sleep
from os import path as os_path
from re import compile as re_compile
from concurrent.futures import ProcessPoolExecutor

from numpy import unique as np_unique
from schedule import every, run_pending
from pandas import DataFrame as pd_DataFrame, concat as pd_concat, read_csv as pd_read_csv


class ReverseIndex:
    """
    Calculate inverse index.
    """

    def __init__(self):
        self.regex_espaco = re_compile(r'\s{2,}')
        self.regex_quebra_linha = re_compile(r'\n+')
        self.regex_palavra = re_compile(r'[a-z]+')
        self.regex_nao_palavra = re_compile(r'[^\w\s]')
        self.stop_words = None

    def get_word_document_id(self, file):
        """
        Get word and document_id.

        Parameters
        ----------
        file : str
            File path.

        Returns
        -------
        df_word : pandas.DataFrame
            Dataset with word and document_id.
        """

        document_id = int(os_path.basename(file))
        with open(file, 'r', encoding='latin-1') as file:
            file_txt = file.read()
        
        file_txt = file_txt.lower()
        file_txt = self.regex_espaco.sub(' ', file_txt)
        file_txt = self.regex_quebra_linha.sub(' ', file_txt)
        file_txt = self.regex_nao_palavra.sub('', file_txt)
        set_words = set(self.regex_palavra.findall(file_txt))
        set_words = set_words - self.stop_words
        list_word = [{'word': word, 'document_id': document_id, } for word in set_words]
        df_word = pd_DataFrame(list_word)
        print(f'{document_id}: {len(set_words)}')
        
        return df_word

    def run_job(self):
        """
        Execute job.

        Parameters
        ----------
        None.

        Returns
        -------
        None
        """

        start_time = time()
        print('***** start *****')

        list_files = glob('dataset/*')
        list_files.sort(key=lambda path: int(os_path.basename(path)))
        print(f'files: {len(list_files)}')

        self.stop_words = set(pd_read_csv('stop_words.csv')['words'])

        time_pool = time()
        with ProcessPoolExecutor(max_workers=2) as pool:
            list_pool = list(pool.map(self.get_word_document_id, list_files))
        print(f'execução_pool: {round(time() - time_pool, 2)}')

        df_word = pd_concat(list_pool)
        print(f'df_word: {df_word.shape}')
        df_word.sort_values('word', inplace=True)

        time_group_by = time()
        df_word = df_word.groupby('word')['document_id'].apply(lambda tmp: np_unique(tmp).tolist()).reset_index()
        print(f'groupby: {round(time() - time_group_by, 2)}')
        print(f'groupby: {df_word.shape}')

        df_word['word_id'] = list(range(1, len(df_word) + 1))

        str_tmp = '\n'.join([str((idx, doc)) for word, doc, idx in df_word.values])
        str_tmp = str_tmp.replace(', ', ',')
        with open('indice_reverso.txt', 'w') as file:
            file.write(str_tmp)

        str_tmp = '\n'.join([f'{word} {idx}' for word, doc, idx in df_word.values])
        with open('dicionario.txt', 'w') as file:
            file.write(str_tmp)

        print(f'tempo_execução: {round(time() - start_time, 2)}')
        print('***** end *****')

        return df_word


if __name__ == '__main__':
    every().day.at('01:00').do(ReverseIndex().run_job).run()
    while True:
        run_pending()
        sleep(1)
