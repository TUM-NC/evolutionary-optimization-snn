import os
import tempfile
import unittest
import warnings

from utility.configuration import Configuration
from utility.validation import is_float, is_int, is_positive


class TestConfiguration(unittest.TestCase):
    def test_cannot_get_not_specified(self):
        configuration = Configuration({"name": "value"})
        self.assertRaises(RuntimeError, configuration.get_config, "name")

    def test_default_value(self):
        configuration = Configuration()
        configuration.set_default_config_value("name", "value", "comment")
        value = configuration.get_config("name")

        self.assertEqual("value", value)

    def test_load_from_yaml(self):
        filename = os.path.join(
            os.path.dirname(__file__), "data/default_config.yaml"
        )
        configuration = Configuration.from_yaml(filename)

        configuration.set_default_config_value(
            "selection_type", "wrong"
        )  # needs to be added, as otherwise can't get config in next step
        selection_type = configuration.get_config("selection_type")
        configuration.set_default_config_value("neuron_mutations", {"a", "b"})
        neuron_mutations = configuration.get_config("neuron_mutations")
        configuration.set_default_config_value(
            "mutation_rates", {"a": "a", "b": "b"}
        )
        mutation_rates = configuration.get_config("mutation_rates")

        self.assertEqual("tournament", selection_type)  # can load string
        self.assertEqual({"threshold"}, neuron_mutations)  # can load a set
        self.assertEqual(0.09, mutation_rates["add_node"])

    def test_default_value_overwrite(self):
        configuration = Configuration({"name": "overwrite"})
        configuration.set_default_config_value("name", "value", "comment")
        value = configuration.get_config("name")

        self.assertEqual("overwrite", value)

    def test_save_default(self):
        configuration = Configuration()
        configuration.set_default_config_value("name", "value", "comment")
        with tempfile.TemporaryFile() as tmp_file:
            configuration.save_yaml(tmp_file)

            tmp_file.seek(0)
            content = tmp_file.read()

        self.assertEqual(b"name: value # comment\n", content)

    def test_save_custom(self):
        configuration = Configuration({"name": "overwrite"})
        configuration.set_default_config_value("name", "value", "comment")
        with tempfile.TemporaryFile() as tmp_file:
            configuration.save_yaml(tmp_file)

            tmp_file.seek(0)
            content = tmp_file.read()

        self.assertEqual(b"name: overwrite # comment\n", content)

    def test_save_filename(self):
        """
        Allow to save as yaml via a filename

        :return:
        """
        configuration = Configuration()
        configuration.set_default_config_value("name", "value", "comment")
        with tempfile.NamedTemporaryFile() as tmp_file:
            filename = tmp_file.name
            configuration.save_yaml(filename)

            content = tmp_file.read()

        self.assertEqual(b"name: value # comment\n", content)

    def test_save_import(self):
        configuration = Configuration({"overwrite": "overwrite"})
        configuration.set_default_config_value("name", "value", "comment")
        configuration.set_default_config_value(
            "overwrite", "default", "comment"
        )
        with tempfile.NamedTemporaryFile() as tmp_file:
            configuration.save_yaml(tmp_file.name)
            tmp_file.seek(0)  # needs to be called, to allow to read from file
            loaded_configuration = Configuration.from_yaml(tmp_file.name)

        # needs to be set again
        loaded_configuration.set_default_config_value(
            "name", "other", "comment"
        )
        loaded_configuration.set_default_config_value(
            "overwrite", "other", "comment"
        )

        self.assertEqual("value", loaded_configuration.get_config("name"))
        self.assertEqual(
            "overwrite", loaded_configuration.get_config("overwrite")
        )

    def test_multiple_same_default_values(self):
        configuration = Configuration()
        configuration.set_default_config_value("name", "value1", "comment")
        with warnings.catch_warnings(record=True) as w:
            configuration.set_default_config_value("name", "value2", "comment")
            self.assertEquals(1, len(w))

    def test_unused_keys_single(self):
        configuration = Configuration({"a": "b"})
        configuration.set_default_config_value("b", "c", "comment")

        unused_keys = configuration.get_unused_keys()
        self.assertEqual(1, len(unused_keys))

    def test_unused_keys_all_used(self):
        configuration = Configuration({"a": "b"})
        configuration.set_default_config_value("a", "b", "comment")

        unused_keys = configuration.get_unused_keys()
        self.assertEqual(0, len(unused_keys))

    def test_validate_config(self):
        configuration = Configuration({"int": 1, "float": 2.2, "positive": 5})
        configuration.set_default_config_value("int", 0, validate=is_int)
        configuration.set_default_config_value("float", 0, validate=is_float)
        configuration.set_default_config_value(
            "positive", 0, validate=is_positive
        )

        self.assertTrue(configuration.validate_config())

        # test with modified configuration
        configuration._configuration["int"] = 5.5
        with warnings.catch_warnings(record=True) as w:
            self.assertFalse(configuration.validate_config())
            self.assertEquals(1, len(w))
