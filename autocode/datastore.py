from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel

from autocode.setting import ApplicationSetting


class OneDatastore:

    def __init__(
            self,
            application_setting: ApplicationSetting,
    ):
        self.path = f"{application_setting.absolute_path}/database.db?cache=shared"
        self.url = f"sqlite:///{self.path}"
        self.engine = create_engine(
            url=self.url,
            isolation_level="SERIALIZABLE"
        )
        SQLModel.metadata.create_all(self.engine)
        sqlite_cache: SQLiteCache = SQLiteCache(database_path=self.path)
        set_llm_cache(sqlite_cache)

    def get_session(self) -> Session:
        session = Session(
            bind=self.engine
        )

        return session
