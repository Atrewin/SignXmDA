cd /home/yejinhui/Projects/SLT/signjoey/external_metrics


hyp=/home/yejinhui/Projects/SLT/training_task/0408_cslt_signKDs_Mixup_ratio0.6_baseline_S2T_seed56_bsz64_drop15_len30_freq100_ratio_1_b4_20_5/best.IT_00017000.BW_03.A_1.dev.txt

ref=/home/yejinhui/Projects/SLT/training_task/0408_cslt_signKDs_Mixup_ratio0.6_baseline_S2T_seed56_bsz64_drop15_len30_freq100_ratio_1_b4_20_5/references.dev.txt

python bleu.py 1 $hyp $ref
python bleu.py 2 $hyp $ref
python bleu.py 3 $hyp $ref
python bleu.py 4 $hyp $ref

