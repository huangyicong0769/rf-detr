from rfdetr import RFDETRClassificationNano


def main():
    model = RFDETRClassificationNano(
        patch_size=14,
        resolution=112,
    )
    model.train(
        dataset_file="coco",
        coco_path="/home/hyc/datasets/imagenet_data/tiny-imagenet-200",
        output_dir="output/cls-tiny",
        num_classes=200,
        epochs=1,
        batch_size=64,
        grad_accum_steps=2,
        use_test_split=False,
        num_workers=8,
        two_stage=False,
        lite_refpoint_refine=False,
        freeze_encoder=True,
        pretrain_weights=None,   # 让它加载 dinov2 预训练
        force_no_pretrain=False, # 确保允许加载
    )


if __name__ == "__main__":
    main()