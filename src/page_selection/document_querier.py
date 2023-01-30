"""
Class to query documents using the INPI API.
"""


class DocumentQuerier:
    """
    Document querier.
    """

    def __init__(self):
        """
        Constructor.
        """

    def query_document(self, siren: str, year: int, save_path: str):
        """
        Query a document using the INPI API. Save at save_path.
        """
        raise NotImplementedError()
