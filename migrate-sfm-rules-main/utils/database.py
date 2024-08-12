"""
    This class is wrapper class to connect to any database. This class exposes functions
    to save and retrieve small or large data.
    df = pd.DataFrame(np.random.random((100,10)))

    engine_args = {"db_type": 'postgresql',
        "db_server": "xxxx",
        "db_port": "xxxx",
        "user": "pxxxxxx",
        "password": "",
       "db_name": "databasename" }
    or engine_args ='sqlite:///my_db.sqlite'

    con = Connector(kwargs)

    con.push_df('tablename',df,is_replace=True,schema='raw',dtype=self.dbcon.df_col_detect(df,True),chunksize=10**5)

    df = con.get_df_table('tablename')
    df = con.get_df_query('SELECT * FROM tablename')

    To work with transaction do below code
    below code will automatically commit once transaction successful or rollback if any screening_exceptions happens.
    It will close connection in any situation.
    con.init_engine()
    with con.engine.begin() as conn:
        response = con.exec_query(_query,_transaction_connection=conn):
        con.push_df('tablename',df,is_replace=True,schema='czuc',con=conn,method='postgrescopy')
    con.engine.dispose()
    if response.cursor:
        result = response.fetchone()
    else:
        result = response.rowcount


    """
import csv
import datetime
import math
import time
from functools import wraps
from io import StringIO

import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text

from utils import logger
from utils.config_loader import config

engine = config["db_configuration"]


class Connector(object):
    def __init__(self, engine_args):
        self.logger = logger.get_logger(__name__)
        self.engine_args = engine_args

        if type(engine_args) == str:
            self.connection_string = engine_args
        else:
            self.connection_string = (
                "{db_type}://{db_user}:{db_password}@{db_server}:{db_port}/{db_name}".format(
                    **self.engine_args
                )
            )

    def init_engine(self, echo=False, **kwargs):
        kwargs["echo"] = echo

        if self.logger:
            self.logger.debug("initiating engine with these options {}".format(kwargs))

        self.engine = create_engine(self.connection_string, **kwargs)

        if self.logger:
            if type(self.engine_args) == str:

                self.logger.info("initiated engine with {}".format(self.engine))
            else:

                self.logger.info(
                    "initiated engine on address: {} with database: {} username: {}".format(
                        self.engine_args["db_server"],
                        self.engine_args["db_name"],
                        self.engine_args["db_user"],
                    )
                )
        # return self.engine

    @staticmethod
    def __remove_non_ascii(value):
        if isinstance(value, (pd.Timestamp, float, int)):
            return value
        if isinstance(value, (datetime.datetime, float, int)):
            return value
        if pd.isnull(value):
            return np.nan
        return "".join(i for i in value if ord(i) < 128)

    def __remove_non_ascii_df(self, df) -> pd.DataFrame:
        for colmn in df.columns:
            df[colmn] = df[colmn].apply(self.__remove_non_ascii)
        return df

    @staticmethod
    def __psql_insert_copy(table, conn, keys, data_iter):
        # gets a DBAPI connection that can provide a cursor
        dbapi_conn = conn.connection
        with dbapi_conn.cursor() as cur:
            s_buf = StringIO()
            writer = csv.writer(s_buf)
            writer.writerows(data_iter)
            s_buf.seek(0)

            columns = ", ".join('"{}"'.format(k) for k in keys)
            if table.schema:
                table_name = "{}.{}".format(table.schema, '"' + table.name + '"')
            else:
                table_name = '"' + table.name + '"'

            sql = "COPY {} ({}) FROM STDIN WITH CSV".format(table_name, columns)

            cur.copy_expert(sql=sql, file=s_buf)

    def __write_df(self, table_name, df, **kwargs):
        """
        This is private function to class. This gets called by other public functions only.
        This function write dataframe to database by calling df.to_sql function
        param: table_name: Table where data to be saved in database
        param: df: This is DataFrame object
        param:**kwargs : This accepts any additional parameters needs to be passed to df.to_sql function
        """
        conn = self.engine
        if "con" in kwargs.keys():
            conn = kwargs["con"]
            kwargs.pop("con", None)

        index = False
        if "index" in kwargs.keys():
            index = kwargs["index"]
            kwargs.pop("index", None)

        df.to_sql(table_name, con=conn, index=index, **kwargs)

        return True

    def __write_split_df(self, table_name, dfs, **kwargs):
        self.__write_df(table_name, dfs[0], **kwargs)

        if self.logger:
            self.logger.info("Pushed {} rows in table : {}".format(len(dfs[0]), table_name))

        kwargs.pop("if_exists", None)
        for df in dfs[1:]:
            self.__write_df(table_name, df, if_exists="append", **kwargs)

            if self.logger:
                self.logger.info("Pushed {} rows in table : {}".format(len(df), table_name))
        return True

    @staticmethod
    def __split_df(df, chunksize):
        chunk_count = int(math.ceil(len(df) / chunksize))
        return np.array_split(df, chunk_count)

    def __df_cleanup(self, df):
        df.columns = df.columns.str.replace("(", "")
        df.columns = df.columns.str.replace(")", "")
        df.columns = df.columns.str.replace("%", "per")
        df.columns = df.columns.str.replace(r"\\t", "")
        df.columns = df.columns.str.replace(r"\\n", "")
        df.columns = df.columns.str.replace(r"\\r", "")
        df.columns = df.columns.str.replace(r"\t", "")
        df.columns = df.columns.str.replace(r"\n", "")
        df.columns = df.columns.str.replace(r"\r", "")

        df = self.__remove_non_ascii_df(df)
        # DF=DF.fillna(np.nan,inplace=True)
        # remove special char from data.
        df.replace(to_replace=[r"\\t|\\n|\\r|'|\t|\n|\r"], value=[""], regex=True, inplace=True)

        return df

    @staticmethod
    def df_col_detect(dfparam: pd.DataFrame, is_all_data_string) -> dict:
        dtypedict = {}

        for i, j in zip(dfparam.columns, dfparam.dtypes):

            if is_all_data_string:
                if i == "itec_date":
                    dtypedict.update({i: sqlalchemy.types.DateTime()})
                else:

                    dtypedict.update({i: sqlalchemy.types.VARCHAR()})
            else:

                if "object" in str(j):
                    dtypedict.update({i: sqlalchemy.types.VARCHAR()})
                if "date" in str(j):
                    dtypedict.update({i: sqlalchemy.types.DateTime()})
                if "float" in str(j):
                    dtypedict.update({i: sqlalchemy.types.Float()})
                if "int" in str(j):
                    dtypedict.update({i: sqlalchemy.types.INT()})
        return dtypedict

    def __check_engine(func):
        """
        This is private Decorating function. Its checks for engine class attribute. If not found then it calls
        init_engine function.
        """

        @wraps(func)
        def inner(*args, **kwargs):
            if not hasattr(args[0], "engine"):

                if "use_batch_mode" in kwargs.keys():
                    # self.logger.info("Using batch mode")
                    args[0].init_engine(use_batch_mode=kwargs["use_batch_mode"])

                else:
                    args[0].init_engine()

            kwargs.pop("use_batch_mode", None)
            return func(*args, **kwargs)

        return inner

    @__check_engine
    def push_df_large(
        self,
        table_name,
        df,
        is_replace=False,
        chunksize=10 ** 5,
        call_df_cleanup=False,
        method=None,
        schema=None,
        **kwargs,
    ):
        """
        This function pushes large datasets to database. It has functionality of spliting dataframes.
        :param table_name: Table to save data.
        :param df: Dataframe to be saved in database.
        :param is_replace: Pass false
        if you want to append data into existing table. Pass True if you want to create or replace table with new
        data.
        :param call_df_cleanup: If True it will call function to clean up df like remove non ascii characters
        from data also remove any special characters like tabs or new line
        :param chunksize: Specify the number of
        rows in each batch to be written at a time. By default, be default only 100000 rows witten in batch.
        :param method :  default None values
        accepted 'postgrescopy' or None or 'multi' Controls the SQL insertion clause used: 'postgrescopy' : When
        insert data in postgresql or postgressql flavour like redshift. This is very fast insert method for Postgres
        database None : Uses standard SQL INSERT clause (one per row). 'multi': Pass multiple values in a single
        INSERT clause.
        :param schema : string, optional
            Specify the schema (if database flavor supports this). If None, use default schema.
        :param: **kwargs: This function calls df.to_sql function internally, any additional parameters need to be
        passed, pass as key and value like
            push_df_large(table_name, df, is_replace=False, chunksize=10 ** 5,schema='raw',
                          dtype=df_col_detect(df,True),use_batch_mode=True) dtype and schema are additional parameters
                          which will be passed to df.to_sql fuction as **kwargs :return : True or False
            con : Assign value only when you want to maintain transaction
                  example :
                    con.init_engine()
                    with con.engine.begin() as conn:
                        response = con.exec_query(_query,_transaction_connection=conn):
                        con.push_df('tablename',df,is_replace=True,schema='czuc',con=conn,method='postgrescopy')
                    con.engine.dispose()
                    if response.cursor:
                        result = response.fetchone()
                    else:
                        result = response.rowcount

            index : bool, default True
            Write DataFrame index as a column. Uses index_label as the column name in the table.

            index_label : string or sequence, default None Column label for index column(s). If None is given (
            default) and index is True, then the index names are used. A sequence should be given if the DataFrame
            uses MultiIndex.

            chunksize : int, optional
            Rows will be written in batches of this size at a time. By default, all rows will be written at once.

            dtype : dict, optional
            Specifying the datatype for columns. The keys should be the column names and the values should be the
            SQLAlchemy types or strings for the sqlite3 legacy mode.
            :return : True or False
            use_batch_mode : Optional. Accept values as True or False
            This is to initiate db engine with use_batch_mode option, if it is not initiated earlier.
            Please read SQLAlchemy's create_engine() documentation where to use use_batch_mode.

        """
        if df.empty:
            raise Exception("Error : Empty DataFrame.")

        if df.columns.duplicated().any():
            raise Exception("Error : Duplicate columns in DataFrame.")

        dispose_engine = True
        if "con" in kwargs.keys():
            dispose_engine = False

        dfsize = chunksize

        if is_replace:
            if_exists = "replace"
        else:
            if_exists = "append"

        kwargs["schema"] = schema

        if call_df_cleanup:
            df = self.__df_cleanup(df)

        table_name = table_name.replace("'", "").replace('"', "").lower()

        status = False

        if dfsize is None:
            dfsize = 10 ** 5

        kwargs.pop("chunksize", None)

        if method is not None:
            if method == "postgrescopy":
                method = self.__psql_insert_copy

            kwargs["method"] = method

        s = time.time()
        if len(df) > dfsize:

            dfs = self.__split_df(df, dfsize)
            status = self.__write_split_df(table_name, dfs, if_exists=if_exists, **kwargs)
            if self.logger:
                self.logger.info(
                    "Total {} rows pushed in table : {} within: {}s".format(
                        len(df), table_name, round(time.time() - s, 4)
                    )
                )

        else:

            status = self.__write_df(
                table_name, df, if_exists=if_exists, chunksize=dfsize, **kwargs
            )
            if self.logger:
                self.logger.info(
                    "Pushed {} rows in table : {} within: {}s".format(
                        len(df), table_name, round(time.time() - s, 4)
                    )
                )

        # if self.logger:
        # self.logger.info('Pushed name: {} dataframe shape: {} within: {}s'.format(table_name,
        # df.shape,
        # round(time.time() - s, 4)))
        if dispose_engine:
            self.__db_dispose()
        return status

    @__check_engine
    def push_df(
        self,
        table_name,
        df,
        is_replace=False,
        call_df_cleanup=False,
        method=None,
        schema=None,
        **kwargs,
    ):
        """
        This function pushes datasets to database.
        :param table_name: Table to save data.
        :param df: Dataframe to be saved in database.
        :param is_replace: Pass false if you want to append data into existing table. Pass True if you want to
                           create or replace table with new data.
        :param call_df_cleanup: If True it will call function to clean up df like remove non ascii characters
                                from data also remove any special characters like tabs or new line
        :param: **kwargs: This function calls df.to_sql function internally, any additional parameters need to be
                          passed, pass as key and value like push_df(table_name, df, is_replace=False,
                          chunksize=10 ** 5,schema='raw',dtype=df_col_detect(df,True),use_batch_mode=True)
                          chunksize ,dtype and schema are additional parameters which will be
                          passed to df.to_sql fuction as **kwargs

        :param method :  default None
            values accepted 'postgrescopy' or None or 'multi'
            Controls the SQL insertion clause used:
            'postgrescopy' : When insert data in postgresql or postgressql flavour like redshift. This is
                             very fast insert method for Postgres database
            None : Uses standard SQL INSERT clause (one per row).
            'multi': Pass multiple values in a single INSERT clause.

        con : Optional, Assign value only when you want to maintain transaction
              example :
                con.init_engine()
                with con.engine.begin() as conn:
                    response = con.exec_query(_query,_transaction_connection=conn):
                    con.push_df('tablename',df,is_replace=True,schema='czuc',con=conn,method='postgrescopy')
                con.engine.dispose()
                if response.cursor:
                    result = response.fetchone()
                else:
                    result = response.rowcount

        :param schema : string, optional
        Specify the schema (if database flavor supports this). If None, use default schema.

        index : bool, default True
        Write DataFrame index as a column. Uses index_label as the column name in the table.

        index_label : string or sequence, default None
        Column label for index column(s). If None is given (default) and index is True, then the
        index names are used. A sequence should be given if the DataFrame uses MultiIndex.

        chunksize : int, optional
        Rows will be written in batches of this size at a time. By default, all rows will be written at once.

        dtype : dict, optional

        use_batch_mode : Optional. Accept values as True or False
        This is to initiate db engine with use_batch_mode option, if it is not initiated earlier.
        Please read SQLAlchemy's create_engine() documentation where to use use_batch_mode.

        """

        if df.empty:
            raise Exception("Error : Empty DataFrame.")

        if df.columns.duplicated().any():
            raise Exception("Error : Duplicate columns in DataFrame.")

        dispose_engine = True
        if "con" in kwargs.keys():
            dispose_engine = False

        if is_replace:
            if_exists = "replace"
        else:
            if_exists = "append"

        kwargs["schema"] = schema

        if call_df_cleanup:
            df = self.__df_cleanup(df)

        # table_name = table_name.replace("'", "").replace('"', "").lower()
        table_name = table_name.replace("'", "").replace('"', "")

        # Format the Dataframe in preparation

        # hit it up

        # status = False

        if method is not None:
            if method == "postgrescopy":
                method = self.__psql_insert_copy

            kwargs["method"] = method

        s = time.time()
        status = self.__write_df(table_name, df, if_exists=if_exists, **kwargs)

        if self.logger:
            self.logger.info(
                "Pushed {} rows in table : {} within: {}s".format(
                    len(df), table_name, round(time.time() - s, 4)
                )
            )
        if dispose_engine:
            self.__db_dispose()
        return status

    @__check_engine
    def get_df_large_table(self, table_name, **kwargs) -> pd.DataFrame:
        """
        This function performance select * from table , with default 100000 rows of batch reading

        Given a table name and a SQLAlchemy connectable, returns a DataFrame. This function does not support
        DBAPI connections.

        Parameters
        table_name : str

            Name of SQL table in database.
        con : SQLAlchemy connectable or str

            A database URI could be provided as as str.
            SQLite DBAPI connection mode not supported.
        schema : str, default None

            Name of SQL schema in database to query (if database flavor
            supports this). Uses default schema if None (default).
        index_col : str or list of str, optional, default: None

            Column(s) to set as index(MultiIndex).
        coerce_float : bool, default True

            Attempts to convert values of non-string, non-numeric objects (like
            decimal.Decimal) to floating point. Can result in loss of Precision.
        parse_dates : list or dict, default None

            - List of column names to parse as dates.
            - Dict of `{column_name: format string}` where format string is
            strftime compatible in case of parsing string times or is one of
            (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of `{column_name: arg dict}`, where the arg dict corresponds
            to the keyword arguments of :func:`pandas.to_datetime`
            Especially useful with databases without native Datetime support,
            such as SQLite.
        columns : list, default None

            List of column names to select from SQL table.
        chunksize : int, default None

            If specified, returns an iterator where `chunksize` is the number of
            rows to include in each chunk.
        Returns
        DataFrame

            A SQL table is returned as two-dimensional data structure with labeled
            axes.
         **kwargs : Any aditional parameter
        """

        s = time.time()
        if "chunksize" not in kwargs.keys():
            kwargs["chunksize"] = 10 ** 5
        dfs = pd.read_sql_table(table_name, self.engine, **kwargs)
        df = pd.DataFrame()
        try:
            df = pd.concat(dfs, axis=0)
        except ValueError:  # No objects to concatenate. dfs is a generator object so has no len() property!
            if self.logger:
                self.logger.warning(
                    "No objects to concatenate on table_name: {}".format(table_name)
                )
            # return None

        length = 0
        if df is not None:
            length = len(df)
        if self.logger:
            self.logger.info(
                "fetched {} rows from  {} within: {}".format(
                    length, table_name, round(time.time() - s, 4)
                )
            )
        self.__db_dispose()
        return df

    @__check_engine
    def get_df_table(self, table_name, **kwargs) -> pd.DataFrame:
        """
        This function performs select * from table , with default brining all rows at once
        Parameters
        table_name : str

            Name of SQL table in database.
        con : SQLAlchemy connectable or str

            A database URI could be provided as as str.
            SQLite DBAPI connection mode not supported.
        schema : str, default None

            Name of SQL schema in database to query (if database flavor
            supports this). Uses default schema if None (default).
        index_col : str or list of str, optional, default: None

            Column(s) to set as index(MultiIndex).
        coerce_float : bool, default True

            Attempts to convert values of non-string, non-numeric objects (like
            decimal.Decimal) to floating point. Can result in loss of Precision.
        parse_dates : list or dict, default None

            - List of column names to parse as dates.
            - Dict of `{column_name: format string}` where format string is
            strftime compatible in case of parsing string times or is one of
            (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of `{column_name: arg dict}`, where the arg dict corresponds
            to the keyword arguments of :func:`pandas.to_datetime`
            Especially useful with databases without native Datetime support,
            such as SQLite.
        columns : list, default None

            List of column names to select from SQL table.
        chunksize : int, default None

            If specified, returns an iterator where `chunksize` is the number of
            rows to include in each chunk.
        Returns
        DataFrame

            A SQL table is returned as two-dimensional data structure with labeled
            axes.
         **kwargs : Any aditional parameter
        """

        s = time.time()
        kwargs.pop("chunksize", None)
        df = pd.read_sql_table(table_name, self.engine, **kwargs)

        length = 0
        if df is not None:
            length = len(df)
        if self.logger:
            self.logger.info(
                "fetched {} rows from  {} within: {}".format(
                    length, table_name, round(time.time() - s, 4)
                )
            )
        self.__db_dispose()
        return df

    @__check_engine
    def exec_query(self, _query, _transaction_connection=None, **kwargs):
        if self.logger:
            self.logger.info('running query: "{}"'.format(_query))
        _query = text(_query)
        result = None
        if _transaction_connection is not None:
            con = _transaction_connection
            result = con.execute(_query)
        else:

            if type(self.engine) == sqlalchemy.engine.base.Engine:
                with self.engine.connect() as con:
                    response = con.execute(_query)
                    if response.cursor:
                        result = response.fetchone()
                    else:
                        result = response.rowcount
            else:
                """Below work around is for SQLite if some how we use native sqlite engine for excute query"""
                result = self.engine.execute(_query)

        if _transaction_connection is None:
            self.__db_dispose()

        return result

    @__check_engine
    def get_df_query(self, _query, sqlite_text_factory=None, **kwargs) -> pd.DataFrame:
        """
        This function performance any query on database. Any DML query
        def read_sql_query(sql, con, index_col=None, coerce_float=True, params=None, parse_dates=None,
                                   chunksize: None=None) Read SQL query into a DataFrame.
        Returns a DataFrame corresponding to the result set of the query string. Optionally provide an index_col
        parameter to use one of the columns as the index, otherwise default integer index will be used.

        Parameters sql : str SQL query or SQLAlchemy Selectable (select or text object) :param sqlite_text_factory:
        This is to work around for sqlite database to avoid screening_exceptions like (sqlite3.OperationalError) Could not
        decode to UTF-8. Pass value as bytes in case of above error. sqlite_text_factory = bytes

            SQL query to be executed.
        con : SQLAlchemy connectable, str, or sqlite3 connection

            Using SQLAlchemy makes it possible to use any DB supported by that
            library. If a DBAPI2 object, only sqlite3 is supported.
        index_col : str or list of str, optional, default: None

            Column(s) to set as index(MultiIndex).
        coerce_float : bool, default True

            Attempts to convert values of non-string, non-numeric objects (like
            decimal.Decimal) to floating point. Useful for SQL result sets.
        params : list, tuple or dict, optional, default: None

            List of parameters to pass to execute method.  The syntax used
            to pass parameters is database driver dependent. Check your
            database driver documentation for which of the five syntax styles,
            described in PEP 249's paramstyle, is supported.
            Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}.
        parse_dates : list or dict, default: None

            - List of column names to parse as dates.
            - Dict of `{column_name: format string}` where format string is
            strftime compatible in case of parsing string times, or is one of
            (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of `{column_name: arg dict}`, where the arg dict corresponds
            to the keyword arguments of :func:`pandas.to_datetime`
            Especially useful with databases without native Datetime support,
            such as SQLite.
        chunksize : int, default None

            If specified, return an iterator where `chunksize` is the number of
            rows to include in each chunk.
        Returns
        DataFrame
        """

        if self.logger:
            self.logger.info('running query: "{}"'.format(_query))

        s = time.time()
        if sqlite_text_factory is None:
            con = self.engine
        else:
            con = self.engine.raw_connection()
            con.connection.text_factory = sqlite_text_factory

        result = pd.read_sql_query(_query, con=con)
        length = 0
        if result is not None:
            length = len(result)
        if self.logger:
            self.logger.info(
                "Finished running query with rows fetched {} within: {}".format(
                    length, round(time.time() - s, 4)
                )
            )

        self.__db_dispose()
        return result

    def __db_dispose(self):
        if type(self.engine) == sqlalchemy.engine.base.Engine:
            self.engine.dispose()


DB_engine = Connector(engine_args=engine)
