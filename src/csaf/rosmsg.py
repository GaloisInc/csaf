""" CSAF ROSMsg Support

Ethan Lew
07/21/20
"""
import os
import pathlib
import importlib.util

import genpy.generator
import genpy.genpy_main


class CsafMsg:
    @staticmethod
    def required_fields():
        """required fields for a CSAF ROSmsg"""
        return ["version_major", "version_minor", "topic", "time"]

    @staticmethod
    def load(msg_fp):
        """load from file pointer"""
        msg_str = msg_fp.read()
        return CsafMsg.loads(msg_str, filename=msg_fp.name)

    @staticmethod
    def loads(msg_str: str, filename=None):
        """load from string"""
        lines = msg_str.splitlines()
        contents = []
        for lidx, l in enumerate(lines):
            ls = l.strip(' \n')
            if len(ls) == 0:
                continue
            lc = ls.split()
            assert len(lc) == 2, f"line number {lidx} fails <type, name> format" + \
                                 ("" if filename is None else f"(filename '{filename}')")
            contents.append(lc)
        return contents

    @classmethod
    def from_msg_file(cls, fname: str):
        with open(fname, 'r') as fp:
            contents = CsafMsg.load(fp)
        return cls(contents)

    def __init__(self, contents):
        self._contents = contents

    @property
    def fields(self):
        return [line[1] for line in self.contents]

    @property
    def fields_no_header(self):
        return [line[1] for line in self.contents if line[1] not in CsafMsg.required_fields()]

    @property
    def contents(self):
        return self._contents


def generate_serializer(msg_filepath: str, output_dir: str, package_name="csaf"):
    """generate a rosmsg class serializer/deserializer given a rosmsg .msg file
    :param msg_filepath: path to .msg file
    :param output_dir: path to place serializer/deserializer
    :param package_name: name of ros package
    :return: None -- asserts that return code of message generator is error free
    """

    # check arguments
    assert os.path.exists(msg_filepath)
    assert os.path.exists(output_dir)

    # see https://github.com/ros/genpy/blob/kinetic-devel/src/genpy/genpy_main.py
    gen = genpy.generator.MsgGenerator()
    retcode = gen.generate_messages(package_name, [msg_filepath], output_dir, {})

    # assert that return from generator is good
    assert retcode == 0

    # import the generated code
    output_python_file = os.path.join(output_dir, f"_{pathlib.Path(msg_filepath).stem}.py")
    assert os.path.exists(output_python_file)
    spec = importlib.util.spec_from_file_location(pathlib.Path(output_python_file).stem,
                                                  output_python_file)
    python_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(python_module)

    # load an instance and return it
    class_name = pathlib.Path(msg_filepath).stem
    class_ = getattr(python_module, class_name)
    return class_
