"""
Class to query documents using the INPI API.
"""
from typing import Tuple
import shutil
import glob
from datetime import datetime
import requests
import zipfile
import tempfile
from .utils import fs
from .bearer_auth import BearerAuth


class DocumentQuerier:
    """
    Document querier.
    """

    def __init__(self, username: str, password: str):
        """
        Constructor.

        Args:
            username (str): Username to use for the INPI API.
            password (str): Password for the INPI API.
        """
        self.username = username
        self.password = password
        self.bearer_auth = BearerAuth(self.get_token())

    def get_token(self):
        """
        Query authentication API to get token.
        """
        json = {"username": self.username, "password": self.password}
        r = requests.post(
            "https://registre-national-entreprises.inpi.fr/api/sso/login",
            json=json,
        )
        return r.json()["token"]

    def update_token(self):
        """
        Update token.
        """
        self.bearer_auth = BearerAuth(self.get_token())

    def query_document(
        self, siren: str, year: int, save_path: str, s3: bool = True
    ):
        """
        Query a document using the INPI API. Save at save_path.

        Args:
            siren (str): Siren identifier.
            year (int): Year for which document is desired.
            save_path (str): Path to save document.
            s3 (bool): True if saving on s3.
        """
        availability, document_id = self.check_document_availability(
            siren, year
        )
        if not availability:
            raise KeyError(f"No document found for {siren} in {year}.")
        self.download_from_id(document_id, save_path, s3)
        return

    def check_document_availability(
        self, siren: str, year: int
    ) -> Tuple[bool, str]:
        """
        Query a document using the INPI API. Save at save_path.

        Args:
            siren (str): Siren identifier.
            year (int): Year for which document is desired.
        """
        document_list = self.list_documents(siren)
        for document_metadata in document_list:
            date_cloture = document_metadata["dateCloture"]
            year_cloture = datetime.strptime(date_cloture, "%Y-%m-%d").year
            if year == year_cloture:
                document_id = document_metadata["id"]
                return True, document_id
        return False, ""

    def download_from_id(
        self, identifier: str, save_path: str, s3: bool = True
    ):
        """
        Download document with given id from the INPI API.

        Args:
            identifier (str): Document id.
            save_path (str): Path to save document.
            s3 (bool): True if saving on s3.
        """
        r = requests.get(
            f"https://registre-national-entreprises.inpi.fr/api/bilans/{identifier}/download",
            auth=self.bearer_auth,
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            # CASE 1: .zip file is downloaded
            try:
                with open(tmpdirname + "tmp.zip", "wb") as f:
                    f.write(r.content)
                with zipfile.ZipFile(tmpdirname + "tmp.zip", "r") as zip_ref:
                    zip_ref.extractall(tmpdirname)
                # Find extracted PDF file
                pdf_files = glob.glob(tmpdirname + "/*.pdf")
                path_to_copy = pdf_files[0]
                if not pdf_files:
                    return ValueError("No PDF file in downloaded archive.")
            # CASE 2: .pdf file is downloaded
            except Exception:
                path_to_copy = tmpdirname + "tmp.pdf"
                with open(path_to_copy, "wb") as f:
                    f.write(r.content)
                
            # Save to `save_path`
            if s3:
                fs.put(path_to_copy, save_path)
            else:
                shutil.copyfile(path_to_copy, save_path)

    def list_documents(self, siren: str):
        """
        Return metadata of all "bilans" documents for the given Siren.

        Args:
            siren (str): Siren identifier.
        """
        r = requests.get(
            f"https://registre-national-entreprises.inpi.fr/api/companies/{siren}/attachments",
            auth=self.bearer_auth,
        )
        print(r.status_code)
        return r.json()["bilans"]
