import os
import pathlib
import importlib.util

import genpy.generator
import genpy.genpy_main


def generate_serializer(msg_filepath, output_dir, package_name="csaf"):
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
    instance = class_()
    return instance



if __name__ == '__main__':
    s0 = generate_serializer("./components/msg/f16plant_state.msg", ".")
    s1 = generate_serializer("./components/msg/f16plant_output.msg", ".")
    print(s0)
    print(s1)
