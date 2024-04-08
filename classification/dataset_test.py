import dataset
from torch.utils.data import DataLoader

def main():
    ds = dataset.PadUfes20("/data/sets/PAD-UFES-20")
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
    for d in loader:
        print(d)

if __name__ == "__main__":
    main()