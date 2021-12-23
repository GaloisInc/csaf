"""
CSAF Library Base

TODO: make a library level logger
TODO: make CsafBase an abstract class, and make validate a mandatory method to implement
"""


class CsafBase:
    """
    CSAF Library Element Base
    """

    def validate(self) -> None:
        """
        validate serves as a duck typing mechanism for the CSAF library to reason whether
        library elements the semantic backplane of the dynamical systems decsribed in
        components

        :return: Nothing, an error is raised if a failure occurs
        """
        raise NotImplementedError

    def check(self, prefix="", raise_error=True) -> bool:
        """
        validation with some additional options

        :param prefix: context to print for the error message
        :param raise_error: whether to log an error and continue, or to raise an exception
        :return:
        """
        try:
            self.validate()
            return True
        except Exception as exc:
            print(prefix, exc)
            if raise_error:
                raise exc
            return False
