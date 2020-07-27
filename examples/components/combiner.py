import toml
import json
import numpy as np

from csaf import message

def main():
    epoch = 0
    fs = 100
    msg_writer = message.Message()

    while True:
        ins = input(f"msg at [t={epoch/fs}]>")
        try:
            msg = json.loads(ins)
        except json.decoder.JSONDecodeError as exc:
            raise Exception(f"input <{ins}> couldn't be interpreted as json! {exc}")

        in_epoch = msg["epoch"]
        epoch = in_epoch
        out = msg["Output"]
        state = msg["State"]

        combine = np.hstack((state, out))
        msg = msg_writer.write_message(epoch/fs, output=combine)

        epoch += 1
        print(msg)

if __name__ == '__main__':
    main()