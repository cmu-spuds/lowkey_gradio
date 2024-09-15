import datasets
import tqdm
from app import protect


def batch_protect(batch: list):
    protect_out = []
    path_out = []
    decoder = datasets.Image()
    for example in tqdm.tqdm(batch):
        protect_out.append(protect(decoder.decode_example(example)))
        path_out.append(example["path"])
    return {"image": protect_out, "path": path_out}


if __name__ == "__main__":
    ds = (
        datasets.load_dataset("logasja/lfw", "pairs", split="test")
        .select_columns("img_0")
        .cast_column("img_0", datasets.Image(decode=False))
        .rename_column("img_0", "image")
    )
    feature_decoder = datasets.Image()
    adv_ds = ds.map(lambda x: batch_protect(x["image"]), batched=True, batch_size=32)
    adv_ds = adv_ds.cast_column("image", datasets.Image())
    adv_ds.save_to_disk("./adversarial_examples")
