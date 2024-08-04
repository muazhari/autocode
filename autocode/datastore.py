from sqlalchemy import create_engine
from sqlmodel import Session

from autocode.setting import ApplicationSetting


class OneDatastore:

    def __init__(
            self,
            application_setting: ApplicationSetting,
    ):
        self.url = f"sqlite:///{application_setting.absolute_path}/database.db?cache=shared"
        self.engine = create_engine(
            url=self.url,
            isolation_level="SERIALIZABLE"
        )

    def get_session(self) -> Session:
        session = Session(
            bind=self.engine
        )

        return session
