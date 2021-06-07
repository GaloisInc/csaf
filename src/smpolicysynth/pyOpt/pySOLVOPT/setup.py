import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('pySOLVOPT', parent_package, top_path)
    config.add_library('solvopt',
                       sources=[os.path.join('source', '*.f')],
                       include_dirs=['source'])
    config.add_extension('solvopt',
                         sources=['source/f2py/solvopt.pyf'],
                         libraries=['solvopt'])
    config.add_data_files('LICENSE', 'README')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())
