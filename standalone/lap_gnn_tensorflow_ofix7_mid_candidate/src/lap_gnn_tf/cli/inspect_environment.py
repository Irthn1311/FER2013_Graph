import json

from lap_gnn_tf.resources import environment_manifest


def main():
    print(json.dumps(environment_manifest(), indent=2, default=str))


if __name__ == "__main__":
    main()

