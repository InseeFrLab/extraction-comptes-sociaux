"""
Utils class for token authentication with requests.
"""
import requests


class BearerAuth(requests.auth.AuthBase):
    """
    BearerAuth class.
    """

    def __init__(self, token):
        """
        Constructor.
        """
        self.token = token

    def __call__(self, r):
        """
        Call method.
        """
        r.headers["authorization"] = "Bearer " + self.token
        return r
