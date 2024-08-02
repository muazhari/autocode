from sqlalchemy import create_engine
from sqlmodel import Session


class OneDatastore:

    def __init__(self):
        self.url = "sqlite:///database.db?cache=shared"
        self.engine = create_engine(
            url=self.url,
            isolation_level="SERIALIZABLE"
        )

    def get_session(self) -> Session:
        session = Session(
            bind=self.engine
        )

        return session
