Hướng dẫn chạy:
1. Tải file json từ lbstudio để vào thư mục dataset
2. chạy file utils/lbstudio_to_json.py => tạo file datasets/data.json
3. chạy file utils/generate_heatmap.py => tạo file heatmap và để trong folder datasets/heatmap
4. chạy file utils/generate_sliding.py => sliding windows ảnh và heatmap và để trong folder datasets/slide
5. chạy file utils/split_train_test.py => để chia tập train/val ở trong folder datasets
6. chạy file train