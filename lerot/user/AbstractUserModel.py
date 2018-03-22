class AbstractUserModel:
    """Defines an abstract base class for user models."""

    def get_clicks(self, result_list, labels, **kwargs):
        raise NotImplementedError("Derived class needs to implement get_clicks.")
