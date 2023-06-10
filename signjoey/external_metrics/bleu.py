import sys
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu


def sacrebleu():
    import sacrebleu

    # 读取参考翻译文件和待评估翻译文件
    ref_file = "/home/yejinhui/Projects/SLT/training_task/0428_S2T_2DExtendGloss_mixSLR_newKD_XmDA_G2T_layer1_3_droall35_50/references.dev.txt"
    hyp_file = "/home/yejinhui/Projects/SLT/training_task/0428_S2T_2DExtendGloss_mixSLR_newKD_XmDA_G2T_layer1_3_droall35_50/best.IT_00774500.BW_02.A_1.dev.txt"

    # 使用SacreBLEU计算BLEU分数
    bleu_score = sacrebleu.corpus_bleu(hyp_file, [ref_file])

    # 打印BLEU分数
    print("BLEU score:", bleu_score.score)

if __name__ == "__main__":
    n = int(sys.argv[1])
    pred_path = sys.argv[2]
    data_path = sys.argv[3]
    weights = [1/n] * n + [0] * (4-n)
    with open(pred_path, "r") as file:
        pred = file.readlines()

    with open(data_path, "r") as file:
        target = file.readlines()

    for index in range(len(target)):
        pred_sentence = pred[index].strip().replace("<pad>", "")
        target_sentence = target[index].strip()

        a = target_sentence.split("|")[-1]

        pred[index] = pred_sentence + " | " + a

    # output_path = pred_path + ".pre_target"
    # with open(output_path, "w") as file:
    #     for line in pred:
    #         file.write(line + "\n")


    pred = [p.replace("<pad>", "").strip() for p in pred]
    pred = [p.split("|")[-1].lower().split() for p in pred]
    target = [[t.split("|")[-1].lower().split()] for t in target]

    score = corpus_bleu(target, pred, weights=weights)
    score = round(score*100, 2)
    print(f"& {score} ", end='')
