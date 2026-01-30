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
        epochs=10,
        batch_size=64,
        grad_accum_steps=2,
        lr_scheduler="step",
        warmup_epochs=3,
        lr_min_factor=0.01,
        weight_decay=0.05,
        drop_path=0.1,
        use_ema=True,
        multi_scale=True,
        expanded_scales=True,
        do_random_resize_via_padding=True,
        use_test_split=False,
        num_workers=12,
        freeze_encoder=True,
        wandb = True,
        project = "test",
        run = "rf-detr-cls-nano-tiny-imagenet_100e",
        pretrain_weights=None,   # 让它加载 dinov2 预训练
        force_no_pretrain=False, # 确保允许加载
    )


if __name__ == "__main__":
    main()