"""Development-only entrypoint; TensorFlow runtime itself remains torch-free."""

from lap_gnn_tf.cli.compare_golden import main


if __name__ == "__main__":
    main()

