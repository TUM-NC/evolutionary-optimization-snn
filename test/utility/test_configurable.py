import unittest
import warnings
from typing import Optional

from utility.configurable import Configurable
from utility.configuration import Configuration


class TestConfigurable(unittest.TestCase):
    def test_multiple_class_same_config(self):
        """
        Test, that multiple implementations of configurable,
        can have same argument
        :return:
        """

        class A(Configurable):
            attribute = "default"

            def set_configurable(self):
                self.add_configurable_attribute("attribute", "an attribute")

        class B(Configurable):
            attribute = "default"

            def set_configurable(self):
                self.add_configurable_attribute("attribute", "an attribute")

        configuration = Configuration({"attribute": "value"})
        a_instance = A(configuration)
        b_instance = B(configuration)

        self.assertEqual("value", a_instance.attribute)
        self.assertEqual("value", b_instance.attribute)

    def test_multiple_class_same_config_different_default(self):
        """
        Default value should be chosen, when nothing is overwriting it
        This could also be different
        :return:
        """

        class A(Configurable):
            attribute = "default_a"

            def set_configurable(self):
                self.add_configurable_attribute("attribute", "an attribute")

        class B(Configurable):
            attribute = "default_b"

            def set_configurable(self):
                self.add_configurable_attribute("attribute", "an attribute")

        configuration = Configuration()
        a_instance = A(configuration)
        with warnings.catch_warnings(record=True) as w:
            b_instance = B(configuration)

        self.assertEqual("default_a", a_instance.attribute)
        self.assertEqual("default_b", b_instance.attribute)
        self.assertEqual(1, len(w))

    def test_default_value_none(self):
        """
        Should be able to set default value to None
        :return:
        """

        class DummyConfigurable(Configurable):
            attribute: Optional[str] = None

            def set_configurable(self):
                self.add_configurable_attribute("attribute", "an attribute")

        configuration1 = Configuration()
        configurable1 = DummyConfigurable(configuration1)

        configuration2 = Configuration({"attribute": "Test"})
        configurable2 = DummyConfigurable(configuration2)

        self.assertEqual(None, configurable1.attribute)
        self.assertEqual("Test", configurable2.attribute)
