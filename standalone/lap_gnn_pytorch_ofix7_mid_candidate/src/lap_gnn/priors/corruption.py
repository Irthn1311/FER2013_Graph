"""Prior corruption is implemented by the locked dataset class."""

from lap_gnn.data.pixel_prior_dataset import D16PixelPriorDataset


def current_probability(dataset: D16PixelPriorDataset) -> float:
    return dataset.current_corruption_probability()
