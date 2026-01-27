from rfdetr import RFDETRSmall

def main():
    model = RFDETRSmall()
    model.train(
        dataset_dir = "/mnt/c/Data/FRS/COCO_Split",
        epochs = 1,
        batch_size = 12,
        grad_accum_steps = 2,
        lr = 1e-4,
        output_dir = "./output/det-test",
        num_workers=4,
    )


if __name__ == "__main__":
    main()
