# QCNN convergence: Classical CNN vs Hybrid vs Full-quantum (MNIST)

Folder này chỉ gồm code model và 1 file chạy.

```
run_convergence.py        # train 3 model + vẽ đường train/validation
models/                   # ClassicalCNN, HybridQCNN, FullQuantumQCNN, tensor-network core, readout, reupload
data/                     # encoding RY (angle) + tiền xử lý 16x16 / grayscale
requirements.txt
```

Cài đặt: `pip install -r requirements.txt` (Python >= 3.10). MNIST tự tải về `mnist_data/`.
Full quantum mô phỏng density matrix: batch 64 cần ~19 GB VRAM (RTX 4090 24 GB đủ), ~15–16 phút/epoch.

## Lệnh chạy (setting tốt nhất cho full quantum; viết đủ cờ cho rõ, các giá trị này cũng là mặc định)

Linux / macOS / Git-Bash:

```bash
python run_convergence.py --models full_quantum --device cuda --seed 0 --split-seed 20260911 \
  --fq-batch-size 64 \
  --fq-pretrain-epochs 20 --fq-pretrain-lr 2e-3 \
  --fq-refine-epochs 5   --fq-refine-lr 2e-4 \
  --fq-epochs 60 --fq-core-lr 1e-4 --fq-readout-lr 1e-3 --fq-patience 10 --fq-reupload \
  --out-dir runs/convergence
```

Windows PowerShell (1 dòng): 

```powershell
python run_convergence.py --models full_quantum --device cuda --seed 0 --split-seed 20260911 --fq-batch-size 64 --fq-pretrain-epochs 20 --fq-pretrain-lr 2e-3 --fq-refine-epochs 5 --fq-refine-lr 2e-4 --fq-epochs 60 --fq-core-lr 1e-4 --fq-readout-lr 1e-3 --fq-patience 10 --fq-reupload --out-dir runs/convergence
```

Cả 3 model rồi vẽ (classical và hybrid chạy vài chục phút, full quantum chạy lâu nhất):

```bash
python run_convergence.py --device cuda --out-dir runs/convergence
```

Chỉ vẽ lại từ file `history_*.csv` đã có (không train): `python run_convergence.py --plot-only --out-dir runs/convergence`
(thêm `--acc-ymin 0` nếu muốn trục accuracy bắt đầu thấp hơn 90%).

Chạy ngắt quãng thì chạy lại đúng lệnh cũ: script tự resume từ epoch hoàn chỉnh cuối cùng (`--no-resume` để bỏ qua).
Kiểm tra nhanh đường ống (vài phút, không cần GPU, dữ liệu giả): `python run_convergence.py --smoke --fake-data --device cpu --out-dir runs/smoke`

## Setting "tốt nhất" của full quantum là gì

Kiến trúc: ảnh 28x28 -> grayscale -> resize 16x16 -> 256 qubit (mã hóa RY) -> 5 khối conv/pool lượng tử -> 8 qubit ->
4 lớp strongly-entangling -> readout Pauli (24 đơn + 28x9 cặp = 276 đặc trưng) -> Linear -> 10 lớp; 3 nhánh độc lập,
trung bình logits; 13.110 tham số (13.134 khi thêm reupload).

Công thức huấn luyện 3 pha (Stage 3D -> 3E -> Stage 9 `B_warm_lr0`), cùng split 50.000 train / 10.000 validation:

| Pha | Việc làm | Optimizer | Chọn checkpoint |
|---|---|---|---|
| 1 pretrain | từ đầu, 20 epoch, batch 64 | Adam lr 2e-3 | val accuracy cao nhất |
| 2 refine | tiếp tục từ pha 1, 5 epoch | Adam MỚI lr 2e-4 | val accuracy cao nhất |
| 3 reupload | thêm `reupload_scale` (=0 lúc đầu, kết quả y hệt pha 2), tối đa 60 epoch | Adam: core 1e-4 / readout 1e-3, ReduceLROnPlateau(0,5; patience 3), dừng sớm patience 10 | val loss thấp nhất |

Kết quả đã đo (1 seed, validation, chưa dùng test): val loss 0,0740, accuracy 97,71% (epoch 57 của pha 3). Mới 1 seed và
đường cong còn đi xuống ở epoch 60, nên khoảng cách với classical chưa kết luận chắc được. Nếu đã có checkpoint
Stage 3E (`best_refined_three_branch.pt`) thì thêm `--fq-warm-start <đường dẫn>` để bỏ qua pha 1-2.

## Hai model còn lại

- Classical CNN: 2 conv (32, 64) + BatchNorm + MLP, 85.258 tham số, Adam 1e-3, batch 128, 20 epoch, ảnh 28x28 nguyên bản.
- Hybrid QCNN: tiền xử lý conv cổ điển -> 8x8 -> lõi TN lượng tử brickwall, 8 qubit cuối, readout Pauli (ZZ);
  3.507 tham số, Adam 1e-3, batch 128, 20 epoch (cấu hình hybrid tốt nhất trong variant search; chỉ chạy sàng lọc 8 epoch).
  Đổi epoch/lr bằng `--classical-*` / `--hybrid-*`.

Các lưu ý: ba model dùng chung split và không dùng tập test; batch size khác nhau (128 so với 64 cho full quantum);
train loss/acc là trung bình chạy trong lúc train (chế độ train), validation tính ở chế độ eval.

## Kết quả sinh ra trong `--out-dir`

`history_<model>.csv`, `convergence_all.png` (3 model trên cùng trục: loss log + accuracy), `convergence_per_model.png`
(mỗi model 1 khung, đánh dấu val loss nhỏ nhất), `summary.json`, cùng các checkpoint `*_best.pt` / `*_last.pt`.
