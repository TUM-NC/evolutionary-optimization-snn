import json


class JsonSerialize:
    """
    Interface for classes, that can be loaded and stored as json
    """

    def to_json_object(self) -> dict:
        """
        Create dict for json dumps

        :return:
        """
        raise NotImplementedError("Please implement this function")

    @classmethod
    def from_json_object(cls, json_object: dict) -> "JsonSerialize":
        """
        Create class from json object

        :param json_object:
        :return:
        """
        raise NotImplementedError("Please implement this function")

    def to_file(self, filename, indent=4):
        """
        Save a class directly to a file

        :param indent: can be a number for spaces or None
        :param filename:
        :return:
        """
        json_object = self.to_json_object()
        text = json.dumps(json_object, indent=indent)

        with open(filename, "w") as f:
            f.write(text)

    @classmethod
    def from_file(cls, filename):
        """
        Read a class from a file

        :param filename:
        :return:
        """
        with open(filename, "r") as f:
            content = f.read()
        json_object = json.loads(content)
        return cls.from_json_object(json_object)
