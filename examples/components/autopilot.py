def main(time=0.0, state=None, input=[0]*4, update=False, output=False):
    """TODO: actually implement the autopilot"""
    if output:
        return [0.0] * 4
    else:
        return


if __name__ == '__main__':
    import fire
    fire.Fire(main)