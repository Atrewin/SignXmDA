


#ChrF2:

output=/home/yejinhui/Projects/SLT/training_task/0502_S2T_KDs2_layer1_3_droall35_50/best.IT_00016000.BW_04.A_-2.dev.txt

ref=/home/yejinhui/Projects/SLT/training_task/0502_S2T_KDs2_layer1_3_droall35_50/references.dev.txt



#cat $ref | sacrebleu $output --metrics chrf --smooth-method exp --smooth-value 0.5 --force

#
sed 's/^.*\|//' $ref > ${ref}_no_id.txt
sed 's/^.*\|//' $output | sacrebleu ${ref}_no_id.txt --metrics chrf --smooth-method exp --smooth-value 0.5 --force



#bleu#
sed 's/^.*\|//' $ref > ${ref}_no_id.txt
sed 's/^.*\|//' $output | sacrebleu ${ref}_no_id.txt



#DE

output=/home/yejinhui/Projects/PGen/G2T/phoenix2014T/results/xx_de_new_88/evaluation/valid_5625.txt

ref=/home/yejinhui/Projects/PGen/G2T/phoenix2014T/results/xx_de_new_88/evaluation/valid.de

hyp=/home/yejinhui/Projects/PGen/G2T/phoenix2014T/results/xx_de_new_88/evaluation/valid_9373.txt
ref=/home/yejinhui/Projects/PGen/G2T/phoenix2014T/results/xx_de_new_88/evaluation/valid.de

output=$hyp

# 过滤参考翻译中的不需要的标识符
sed -E 's/dev\/[0-9A-Za-z_-]+\|//g' $ref > ${ref}_no_id.txt
sed -E 's/dev\/[0-9A-Za-z_-]+\|//g' $output | sacrebleu ${ref}_no_id.txt --metrics chrf --smooth-method exp --smooth-value 0.5 --force

# 计算BLEU分数
sed -E 's/dev\/[0-9A-Za-z_-]+\|//g' $ref > ${ref}_no_id.txt
sed -E 's/dev\/[0-9A-Za-z_-]+\|//g' $output | sacrebleu ${ref}_no_id.txt
